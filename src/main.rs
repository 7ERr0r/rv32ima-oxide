// Copyright 2022 Charles Lohr, you may use this file or any portions herein under any of the BSD, MIT, or CC0 licenses.

use std::{io::Write, num::Wrapping, time::Duration};

/*
    To use mini-rv32ima.h for the bare minimum, the following:
    #define MINI_RV32_RAM_SIZE ram_amt
    #define MINIRV32_IMPLEMENTATION
    #include "mini-rv32ima.h"
    Though, that's not _that_ interesting. You probably want I/O!
    Notes:
        * There is a dedicated CLNT at 0x10000000.
        * There is free MMIO from there to 0x12000000.
        * You can put things like a UART, or whatever there.
        * Feel free to override any of the functionality with macros.
*/

/// UNSAFE actually
///
/// returns another mutable pointer to the same memory
///
/// but memory there is just a bunch of u32 or u8 so who cares
///
/// ```
/// // safe version                                                \/ - note lifetime here
/// pub fn ref_core_last_bytes<'a>(ram_image: &'a mut RVImage) -> &'a mut MiniRV32IMAState {
/// ```
///
pub fn ref_core_last_bytes<'a, 'b>(ram_image: &'a mut RVImage) -> &'b mut MiniRV32IMAState
where
    'b: 'a,
{
    let core_len = core::mem::size_of::<MiniRV32IMAState>();
    let len = ram_image.image.len();
    let offset = len - core_len;
    let slice = &mut ram_image.image[offset..offset + core_len];

    let ptr = slice.as_mut_ptr();
    let ptr = ptr as *mut MiniRV32IMAState;
    unsafe {
        let ptr = ptr.as_mut().expect("*mut MiniRV32IMAState .as_mut()");
        ptr
    }
}

static DEBUG_INSTR: bool = false;

pub fn main() {
    let ram_amt: u32 = MINI_RV32_RAM_SIZE;
    let mut instct: u64 = 13000_000_000 as u64;
    //let show_help = 0;
    let time_divisor = 1;
    let fixed_time_update = false;
    let do_sleep = true;
    let single_step = false;

    

    let mut ram_image = RVImage {
        image: vec![0; ram_amt as usize],
    };

    {
        let file_image = include_bytes!("DownloadedImage");
        let im = &mut ram_image.image[..file_image.len()];
        im.copy_from_slice(file_image);
    }

    let opt_dtb_bytes = Some(DEFAULT64MBDTB);
    let mut dtb_ptr: u32 = 0;
    if let Some(dbt_bytes) = opt_dtb_bytes {
        //include_bytes!("sixtyfourmb.dtb");
        let dtb_len = dbt_bytes.len();
        let core_len = core::mem::size_of::<MiniRV32IMAState>();
        let dtb_offset = ram_amt as usize - dtb_len - core_len;
        let im = &mut ram_image.image[dtb_offset..dtb_offset + dtb_len];
        im.copy_from_slice(dbt_bytes);
        // }

        // if let Some(dbt_bytes) = opt_dtb_bytes {
        // Update system ram size in DTB (but if and only if we're using the default DTB)
        // Warning - this will need to be updated if the skeleton DTB is ever modified.
        //let dtb_offset = (uint32_t*)(ram_image + dtb_ptr);
        let value_offset = dtb_offset as u32 + 0x13c;
        if ram_image.load32(value_offset) == 0x00c0ff03 {
            let validram = dtb_offset as u32;
            ram_image.store32be(value_offset, validram);
        }
        dtb_ptr = dtb_offset as u32;
    }

    let mut handler = RVHandlerImpl::default();

    let mut proc_state_obj;
    let mut proc_state: &mut MiniRV32IMAState;
    if true {
        // The core lives on the heap
        proc_state_obj = Box::new(MiniRV32IMAState::default());
        proc_state = &mut proc_state_obj;
    } else {
        // The core lives at the end of RAM.
        proc_state = ref_core_last_bytes(&mut ram_image);
    }
    let reg_a1_ram_size = if opt_dtb_bytes.is_some() {
        dtb_ptr + MINIRV32_RAM_IMAGE_OFFSET
    } else {
        0
    };
    proc_state.pc = MINIRV32_RAM_IMAGE_OFFSET;
    proc_state.regs[10] = 0x00; //hart ID
    proc_state.regs[11] = reg_a1_ram_size; //dtb_pa (Must be valid pointer) (Should be pointer to dtb)

    proc_state.extraflags |= 3; // Machine-mode.

    // Image is loaded.
    let mut last_time: u64 = if fixed_time_update {
        0
    } else {
        time_now_micros() / time_divisor
    };
    let instrs_per_flip = if single_step { 1 } else { 1024 };
    let mut rt: u64 = 0;
    while rt < instct + 1 {
        let elapsed_us: u64;
        if fixed_time_update {
            elapsed_us = proc_state.cycle() / time_divisor - last_time;
        } else {
            elapsed_us = time_now_micros() / time_divisor - last_time;
        }
        last_time += elapsed_us;

        if single_step {
            dump_state(&proc_state, &ram_image, &ram_amt);
        }
        let ret = mini_rv32_ima_step(
            &mut proc_state,
            &mut ram_image,
            &mut handler,
            0,
            elapsed_us as u32,
            instrs_per_flip,
        ); // Execute upto 1024 cycles before breaking out.
        let ret = ret.unwrap_or_else(|ret| ret);
        match ret {
            0 => {}
            1 => {
                if do_sleep {
                    mini_sleep();
                    proc_state.set_cycle(proc_state.cycle() + instrs_per_flip as u64);
                }
            }
            3 => {
                instct = 0;
            }
            0x7777 => {
                //goto restart;	//syscon code for restart
            }
            0x5555 => {
                println!("POWEROFF@0x{:08x}0x{:08x}", proc_state.cycleh, proc_state.cyclel);
                return; //syscon code for power-off
            }
            _default => {
                println!("Unknown failure");
                break;
            }
        }

        rt += instrs_per_flip as u64;
    }
    println!("end of loop");

    dump_state(&proc_state, &ram_image, &ram_amt);
}



pub fn dump_state(core: &MiniRV32IMAState, image: &RVImage, ram_amt: &u32) {
    let pc = core.pc;
    let pc_offset = pc.wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);
    let ir;

    print!("PC: {:08x} ", pc);
    if pc_offset <= ram_amt - 4 {
        ir = image.load32(pc_offset);
        print!("[0x{:08x}] ", ir);
    } else {
        print!("[xxxxxxxxxx] ");
    }
    let regs = &core.regs;
    print!(
        "Z:{:08x} ra:{:08x} sp:{:08x} gp:{:08x} tp:{:08x} t0:{:08x} t1:{:08x} t2:{:08x} ",
        regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7],
    );
    print!(
        "s0:{:08x} s1:{:08x} a0:{:08x} a1:{:08x} a2:{:08x} a3:{:08x} a4:{:08x} a5:{:08x} ",
        regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15]
    );

    print!(
        "a6:{:08x} a7:{:08x} s2:{:08x} s3:{:08x} s4:{:08x} s5:{:08x} s6:{:08x} s7:{:08x}",
        regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23],
    );
    print!(
        "s8:{:08x} s9:{:08x} s10:{:08x} s11:{:08x} t3:{:08x} t4:{:08x} t5:{:08x} t6:{:08x}\n",
        regs[24], regs[25], regs[26], regs[27], regs[28], regs[29], regs[30], regs[31]
    );
}

pub fn time_now_micros() -> u64 {
    let start = std::time::SystemTime::now();
    let since_the_epoch = start
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_micros() as u64
}

static MINI_RV32_RAM_SIZE: u32 = 1024 * 1024 * 64;
static MINIRV32_RAM_IMAGE_OFFSET: u32 = 0x80000000;

pub trait RVHandler {
    fn postexec(&mut self, pc: u32, ir: u32, retval: &mut u32) -> Result<i32, i32>;
    fn handle_mem_store_control(&mut self, addy: u32, rs2: u32) -> u32;
    fn handle_mem_load_control(&mut self, addy: u32) -> u32;
    fn othercsr_write(&mut self, image: &RVImage, csrno: u32, writeval: u32);
    fn othercsr_read(&mut self, csrno: u32, rval: u32);
}

#[derive(Default)]
pub struct RVHandlerImpl {
    pub uart_buf: Vec<u8>,
}

impl RVHandlerImpl {
    fn handle_exception(&mut self, _ir: u32, code: u32) -> u32 {
        // Weird opcode emitted by duktape on exit.
        if code == 3 {
            // Could handle other opcodes here.
        }
        code
    }
    
}
impl RVHandler for RVHandlerImpl {

    fn postexec(&mut self, _pc: u32, ir: u32, retval: &mut u32) -> Result<i32, i32> {
        let fail_on_all_faults = false;
        if *retval > 0 {
            if fail_on_all_faults {
                println!("FAULT retval: trap=={} (signed {})", retval, *retval as i32);
                return Err(3);
            } else {
                //retval = retval;
                *retval = self.handle_exception(ir, *retval);
            }
        }
        return Ok(0);
    }

    fn handle_mem_store_control(&mut self, addy: u32, val: u32) -> u32 {
        if addy == 0x10000000 {
            //UART 8250 / 16550 Data Buffer
    
            let character = val as u8;
            self.uart_buf.push(character);
            if character == '\n' as u8 {
                let mut out = std::io::stdout();
                out.write("uart: ".as_bytes()).unwrap();
                out.write(&self.uart_buf).unwrap();
                self.uart_buf.clear();
            }
    
            // std::io::stdout().flush().unwrap();
            //println!("UART: {}", val as u8 as char);
            //fflush( stdout );
        }
        return 0;
    }

    fn handle_mem_load_control(&mut self, addy: u32) -> u32 {
        // Emulating a 8250 / 16550 UART
        if addy == 0x10000005 {
            let hitbit = if is_keyboard_hit() { 1 } else { 0 };
            return 0x60 | hitbit;
        } else if addy == 0x10000000 && is_keyboard_hit() {
            return read_keyboard_byte();
        }
        return 0
    }

    fn othercsr_write(&mut self, image: &RVImage, csrno: u32, value: u32) {
        if csrno == 0x136 {
            println!("{}", value);
        }
        if csrno == 0x137 {
            println!("0x{:#08x}", value);
        } else if csrno == 0x138 {
            //Print "string"
            let ptr_start = value.wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);
            let mut ptr_end = ptr_start;
            let ram_amt = image.image.len() as u32;
            if ptr_start >= ram_amt {
                println!("DEBUG PASSED INVALID PTR (0x{:#08x})", value);
            }
            while let Some(&char) = image.image.get(ptr_end as usize) {
                if char == 0 {
                    break;
                }
                ptr_end += 1;
            }
            if ptr_end != ptr_start {
                let slice = &image.image[ptr_start as usize..ptr_end as usize];
                let s = String::from_utf8_lossy(&slice);
                println!("{}", s);
            }
        }
    }

    fn othercsr_read(&mut self, _csrno: u32, _rval: u32) {
        
    }
}

// fn minirv32_postexec(pc: u32, ir: u32, retval: &mut u32) -> Result<i32, i32> {
//     minirv32_postexec_impl(pc, ir, retval)
// }

// fn minirv32_handle_mem_store_control(addy: u32, rs2: u32) {
//     handle_control_store_impl(addy, rs2);
// }

// fn minirv32_handle_mem_load_control(addy: u32, rval: &mut u32) {
//     *rval = handle_control_load_impl(addy);
// }

// fn minirv32_othercsr_write(image: &RVImage, csrno: u32, writeval: u32) {
//     handle_other_csr_write_impl(image, csrno as u16, writeval);
// }








static is_eofd: bool = false;
fn read_keyboard_byte() -> u32 {
    if is_eofd {
        return 0xffffffff;
    }
    let rxchar = 0;
    //let rread = read(fileno(stdin), (char*)&rxchar, 1);
    let rread = 0;
    if rread > 0 {
        // Tricky: getchar can't be used with arrow keys.
        return rxchar;
    } else {
        return -1 as i32 as u32;
    }
}

fn is_keyboard_hit() -> bool {
    if is_eofd {
        return false;
    }
    //let byteswaiting;
    //ioctl(0, FIONREAD, &byteswaiting);
    //if( !byteswaiting && write( fileno(stdin), 0, 0 ) != 0 ) { is_eofd = 1; return -1; } // Is end-of-file for
    //return !!byteswaiting;

    false
}

fn mini_sleep() {
    std::thread::sleep(Duration::from_micros(500));
}

// As a note: We quouple-ify these, because in HLSL, we will be operating with
// uint4's.  We are going to uint4 data to/from system RAM.
//
// We're going to try to keep the full processor state to 12 x uint4.

#[derive(Clone, Default)]
pub struct MiniRV32IMAState {
    pub regs: [u32; 32],

    pub pc: u32,
    pub mstatus: u32,
    pub cyclel: u32,
    pub cycleh: u32,

    pub timerl: u32,
    pub timerh: u32,
    pub timermatchl: u32,
    pub timermatchh: u32,

    pub mscratch: u32,
    pub mtvec: u32,
    pub mie: u32,
    pub mip: u32,

    pub mepc: u32,
    pub mtval: u32,
    pub mcause: u32,

    // Note: only a few bits are used.  (Machine = 3, User = 0)
    // Bits 0..1 = privilege.
    // Bit 2 = WFI (Wait for interrupt)
    pub extraflags: u32,
}
impl MiniRV32IMAState {
    pub fn reg(&self, reg_index: u32) -> u32 {
        self.regs[reg_index as usize]
    }
    pub fn regset(&mut self, reg_index: u32, value: u32) {
        self.regs[reg_index as usize] = value
    }
    pub fn cycle(&self) -> u64 {
        self.cyclel as u64
    }
    pub fn set_cycle(&mut self, cycle: u64) {
        self.cyclel = cycle as u32
    }
}

pub struct RVImage {
    image: Vec<u8>,
}
impl RVImage {
    pub fn load32(&self, offset: u32) -> u32 {
        let ofs = offset as usize;
        let slice = &self.image[ofs..ofs + 4];

        // Always safe, since slice above is checked
        <u32>::from_le_bytes(unsafe {
            *(slice.as_ptr() as *const [u8; core::mem::size_of::<u32>()])
        })
    }
    pub fn load16(&self, offset: u32) -> u16 {
        let ofs = offset as usize;
        let slice = &self.image[ofs..ofs + 2];

        // Always safe, since slice above is checked
        <u16>::from_le_bytes(unsafe {
            *(slice.as_ptr() as *const [u8; core::mem::size_of::<u16>()])
        })
    }
    pub fn load8(&self, offset: u32) -> u8 {
        self.image[offset as usize]
    }
    pub fn store32(&mut self, offset: u32, val: u32) {
        let ofs = offset as usize;
        let slice = &mut self.image[ofs..ofs + 4];

        // Always safe, since slice above is checked
        unsafe {
            let ptr = slice.as_mut_ptr() as *mut [u8; core::mem::size_of::<u32>()];
            *ptr = val.to_le_bytes();
        };
    }
    pub fn store32be(&mut self, offset: u32, val: u32) {
        let ofs = offset as usize;
        let slice = &mut self.image[ofs..ofs + 4];

        // Always safe, since slice above is checked
        unsafe {
            let ptr = slice.as_mut_ptr() as *mut [u8; core::mem::size_of::<u32>()];
            *ptr = val.to_be_bytes();
        };
    }
    pub fn store16(&mut self, offset: u32, val: u16) {
        let ofs = offset as usize;
        let slice = &mut self.image[ofs..ofs + 2];

        // Always safe, since slice above is checked
        unsafe {
            let ptr = slice.as_mut_ptr() as *mut [u8; core::mem::size_of::<u16>()];
            *ptr = val.to_le_bytes();
        };
    }
    pub fn store8(&mut self, offset: u32, val: u8) {
        self.image[offset as usize] = val
    }
}

pub fn mini_rv32_ima_step<H: RVHandler>(
    state: &mut MiniRV32IMAState,
    image: &mut RVImage,
    handler: &mut H,
    _v_proc_address: u32,
    elapsed_us: u32,
    count: u32,
) -> Result<i32, i32> {
    let new_timer: u32 = state.timerl + elapsed_us;
    if new_timer < state.timerl {
        state.timerh += 1;
    }
    state.timerl = new_timer;

    // Handle Timer interrupt.
    if (state.timerh > state.timermatchh
        || (state.timerh == state.timermatchh && state.timerl > state.timermatchl))
        && (state.timermatchh != 0 || state.timermatchl != 0)
    {
        state.extraflags &= !4; // Clear WFI
        state.mip |= 1 << 7; //MTIP of MIP // https://stackoverflow.com/a/61916199/2926815  Fire interrupt.
    } else {
        state.mip &= !(1 << 7);
    }
    // If WFI, don't run processor.
    if (state.extraflags & 4) != 0 {
        return Ok(1);
    }

    for _icount in 0..count {
        let mut ir = 0;
        let mut trap = 0; // If positive, is a trap or interrupt.  If negative, is fatal error.
        let mut rval = 0;

        // Increment both wall-clock and instruction count time.  (NOTE: Not strictly needed to run Linux)
        state.cyclel += 1;
        if state.cyclel == 0 {
            state.cycleh += 1;
        }

        let mut pc: u32 = state.pc;
        let ofs_pc: u32 = pc.wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);

        if ofs_pc >= MINI_RV32_RAM_SIZE {
            trap = 1 + 1; // Handle access violation on instruction read.
        } else if ofs_pc & 3 != 0 {
            trap = 1 + 0; //Handle PC-misaligned access
        } else {
            ir = image.load32(ofs_pc);
            if DEBUG_INSTR {
                println!("IR 0x{:08x}", ir);
            }
            let mut rdid: u32 = (ir >> 7) & 0x1f;

            match ir & 0x7f {
                0b0110111 => {
                    // LUI

                    rval = ir & 0xfffff000;
                }
                0b0010111 => {
                    // AUIPC
                    rval = pc.wrapping_add(ir & 0xfffff000);
                }
                0b1101111 => {
                    // JAL

                    let mut reladdy: i32 = (((ir & 0x80000000) >> 11)
                        | ((ir & 0x7fe00000) >> 20)
                        | ((ir & 0x00100000) >> 9)
                        | (ir & 0x000ff000)) as i32;

                    if DEBUG_INSTR {
                        //println!("JAL 0x{:08x}", reladdy);
                    }
                    if reladdy & 0x00100000 != 0 {
                        reladdy |= 0xffe00000 as u32 as i32; // Sign extension.
                    }
                    rval = pc + 4;
                    //pc = pc + reladdy - 4;
                    pc = (pc as i32 + reladdy as i32).wrapping_sub(4) as u32;
                }
                0b1100111 => {
                    // JALR

                    let imm: u32 = ir >> 20;
                    let extension = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm_se: i32 = (imm | extension) as i32;
                    rval = pc + 4;
                    let reg: i32 = state.reg((ir >> 15) & 0x1f) as i32;
                    let new_pc: i32 = (reg.wrapping_add(imm_se)) & !1;
                    pc = new_pc.wrapping_sub(4) as u32;
                }
                0b1100011 => {
                    // Branch
                    // uint32_t immm4 = ((ir & 0xf00)>>7) | ((ir & 0x7e000000)>>20) | ((ir & 0x80) << 4) | ((ir >> 31)<<12);
                    // if( immm4 & 0x1000 ) immm4 |= 0xffffe000;
                    // int32_t rs1 = REG((ir >> 15) & 0x1f);
                    // int32_t rs2 = REG((ir >> 20) & 0x1f);
                    // immm4 = pc + immm4 - 4;

                    let mut immm4: u32 = ((ir & 0xf00) >> 7)
                        | ((ir & 0x7e000000) >> 20)
                        | ((ir & 0x80) << 4)
                        | ((ir >> 31) << 12);
                    if immm4 & 0x1000 != 0 {
                        immm4 |= 0xffffe000;
                    }
                    let rs1: i32 = state.reg((ir >> 15) & 0x1f) as i32;
                    let rs2: i32 = state.reg((ir >> 20) & 0x1f) as i32;

                    immm4 = (Wrapping(pc) + Wrapping(immm4) - Wrapping(4)).0;

                    if DEBUG_INSTR {
                        println!("BRANCH 0x{:08x} 0x{:08x} 0x{:08x}", rs1, rs2, immm4);
                    }
                    //immm4 = pc + immm4 - 4;
                    rdid = 0;
                    match (ir >> 12) & 0x7 {
                        // BEQ, BNE, BLT, BGE, BLTU, BGEU
                        0b000 => {
                            if rs1 == rs2 {
                                pc = immm4;
                            }
                        }
                        0b001 => {
                            if rs1 != rs2 {
                                pc = immm4;
                            }
                        }
                        0b100 => {
                            if rs1 < rs2 {
                                pc = immm4;
                            }
                        }
                        0b101 => {
                            //BGE
                            if rs1 >= rs2 {
                                pc = immm4;
                            }
                        }
                        0b110 => {
                            //BLTU
                            if (rs1 as u32) < rs2 as u32 {
                                pc = immm4;
                            }
                        }
                        0b111 => {
                            //BGEU
                            if (rs1 as u32) >= rs2 as u32 {
                                pc = immm4;
                            }
                        }
                        _default => {
                            trap = 2 + 1;
                        }
                    }
                }
                0b0000011 => {
                    // Load

                    let rs1: u32 = state.reg((ir >> 15) & 0x1f);
                    let imm: u32 = ir >> 20;
                    let extension: u32 = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm_se: i32 = (imm | extension) as i32;
                    let mut rsval: u32 = (rs1 as i32).wrapping_add(imm_se) as u32;

                    rsval = rsval.wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);
                    if rsval >= MINI_RV32_RAM_SIZE - 3 {
                        rsval -= MINIRV32_RAM_IMAGE_OFFSET;
                        if rsval >= 0x10000000 && rsval < 0x12000000 {
                            // UART, CLNT

                            if rsval == 0x1100bffc {
                                // https://chromitem-soc.readthedocs.io/en/latest/clint.html
                                rval = state.timerh;
                            } else if rsval == 0x1100bff8 {
                                rval = state.timerl;
                            } else {
                                rval = handler.handle_mem_load_control(rsval);
                            }
                        } else {
                            trap = 5 + 1;
                            rval = rsval;
                        }
                    } else {
                        match (ir >> 12) & 0b111 {
                            //LB, LH, LW, LBU, LHU
                            0b000 => {
                                rval = image.load8(rsval) as i8 as u32;
                            }
                            0b001 => {
                                rval = image.load16(rsval) as i16 as u32;
                            }
                            0b010 => {
                                rval = image.load32(rsval);
                                if DEBUG_INSTR {
                                    println!("load32 image[0x{:08x}] = 0x{:08x}", rsval, rval);
                                }
                            }
                            0b100 => {
                                rval = image.load8(rsval) as u32;
                            }
                            0b101 => {
                                rval = image.load16(rsval) as u32;
                            }
                            _default => {
                                trap = 2 + 1;
                            }
                        }
                    }
                }
                0b0100011 => {
                    // Store

                    let rs1: u32 = state.reg((ir >> 15) & 0b11111);
                    let rs2: u32 = state.reg((ir >> 20) & 0b11111);
                    let mut addy: u32 = ((ir >> 7) & 0x1f) | ((ir & 0xfe000000) >> 20);
                    if addy & 0x800 != 0 {
                        addy |= 0xfffff000;
                    }
                    addy = addy
                        .wrapping_add(rs1)
                        .wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);

                    if DEBUG_INSTR {
                        println!("STORE 0x{:08x} 0x{:08x} 0x{:08x}", rs1, rs2, addy);
                    }
                    rdid = 0;

                    if addy > MINI_RV32_RAM_SIZE - 4 {
                        addy = addy.wrapping_sub(MINIRV32_RAM_IMAGE_OFFSET);
                        if addy >= 0x10000000 && addy < 0x12000000 {
                            // Should be stuff like SYSCON, 8250, CLNT
                            if addy == 0x11004004 {
                                //CLNT
                                state.timermatchh = rs2;
                            } else if addy == 0x11004000 {
                                //CLNT
                                state.timermatchl = rs2;
                            } else if addy == 0x11100000 {
                                //SYSCON (reboot, poweroff, etc.)

                                state.pc = state.pc + 4;
                                return Ok(rs2 as i32); // NOTE: PC will be PC of Syscon.
                            } else {
                                handler.handle_mem_store_control(addy, rs2);
                            }
                        } else {
                            trap = 7 + 1; // Store access fault.
                            rval = addy.wrapping_add(MINIRV32_RAM_IMAGE_OFFSET);
                        }
                    } else {
                        match (ir >> 12) & 0b111 {
                            //SB, SH, SW
                            0b000 => {
                                image.store8(addy, rs2 as u8);
                            }
                            0b001 => {
                                image.store16(addy, rs2 as u16);
                            }
                            0b010 => {
                                image.store32(addy, rs2);
                            }
                            _default => {
                                trap = 2 + 1;
                            }
                        }
                    }
                }

                0b0110011 | 0b0010011 => {
                    // Op // Op-immediate

                    let imm = ir >> 20;
                    let extenstion = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm = imm | extenstion;
                    let rs1 = state.reg((ir >> 15) & 0x1f);
                    let is_reg = (ir & 0b100000) != 0;
                    let rs2 = if is_reg { state.reg(imm & 0x1f) } else { imm };

                    if DEBUG_INSTR {
                        println!(
                            "OP-IM 0x{:08x} 0x{:08x} 0x{:08x}",
                            rs1,
                            rs2,
                            if is_reg { 1 } else { 0 }
                        );
                    }

                    if is_reg && (ir & 0x02000000 != 0) {
                        match (ir >> 12) & 7 {
                            //0x02000000 = RV32M
                            0b000 => {
                                rval = rs1.wrapping_mul(rs2);
                            } // MUL
                            0b001 => {
                                rval = (((rs1 as i32 as i64) * (rs2 as i32 as i64)) >> 32) as u32;
                            } // MULH
                            0b010 => {
                                rval = (((rs1 as i32 as i64) * (rs2 as u64 as i64)) >> 32) as u32;
                            } // MULHSU
                            0b011 => {
                                rval = (((rs1 as u64) * (rs2 as u64)) >> 32) as u32;
                            } // MULHU
                            0b100 => {
                                if rs2 == 0 {
                                    rval = -1 as i32 as u32;
                                } else {
                                    rval = ((rs1 as i32) / (rs2 as i32)) as u32;
                                }
                            } // DIV
                            0b101 => {
                                if rs2 == 0 {
                                    rval = 0xffffffff;
                                } else {
                                    rval = rs1 / rs2;
                                }
                            } // DIVU
                            0b110 => {
                                if rs2 == 0 {
                                    rval = rs1;
                                } else {
                                    rval = ((rs1 as i32) % (rs2 as i32)) as u32;
                                }
                            } // REM
                            0b111 => {
                                if rs2 == 0 {
                                    rval = rs1;
                                } else {
                                    rval = rs1 % rs2;
                                }
                            } // REMU
                            _default => {}
                        }
                    } else {
                        match (ir >> 12) & 0b111 {
                            // These could be either op-immediate or op commands.  Be careful.
                            0b000 => {
                                // addi
                                rval = if is_reg && (ir & 0x40000000) != 0 {
                                    rs1.wrapping_sub(rs2)
                                } else {
                                    rs1.wrapping_add(rs2)
                                };
                            }
                            0b001 => {
                                rval = rs1.wrapping_shl(rs2);
                            }
                            0b010 => {
                                rval = ((rs1 as i32) < (rs2 as i32)) as u32;
                            }
                            0b011 => {
                                rval = (rs1 < rs2) as u32;
                            }
                            0b100 => {
                                rval = rs1 ^ rs2;
                            }
                            0b101 => {
                                rval = if ir & 0x40000000 != 0 {
                                    ((rs1 as i32).wrapping_shr(rs2)) as u32
                                } else {
                                    rs1.wrapping_shr(rs2)
                                };
                            }
                            0b110 => {
                                rval = rs1 | rs2;
                            }
                            0b111 => {
                                rval = rs1 & rs2;
                            }
                            _default => {}
                        }
                    }
                }
                0b0001111 => {
                    rdid = 0; // fencetype = (ir >> 12) & 0b111; We ignore fences in this impl.
                }
                0b1110011 => {
                    // Zifencei+Zicsr

                    let csrno = ir >> 20;
                    let microop = (ir >> 12) & 0b111;
                    if (microop & 3) != 0 {
                        // It's a Zicsr function.

                        let rs1imm = (ir >> 15) & 0x1f;
                        let rs1 = state.reg(rs1imm);
                        let mut writeval = rs1;

                        // https://raw.githubusercontent.com/riscv/virtual-memory/main/specs/663-Svpbmt.pdf
                        // Generally, support for Zicsr
                        match csrno {
                            0x340 => rval = state.mscratch,
                            0x305 => rval = state.mtvec,
                            0x304 => rval = state.mie,
                            0xC00 => rval = state.cyclel,
                            0x344 => rval = state.mip,
                            0x341 => rval = state.mepc,
                            0x300 => rval = state.mstatus, //mstatus
                            0x342 => rval = state.mcause,
                            0x343 => rval = state.mtval,
                            0xf11 => {
                                rval = 0xff0ff0ff;
                            } //mvendorid
                            0x301 => {
                                rval = 0x40401101;
                            } //misa (XLEN=32, IMA+X)
                            //0x3B0: rval = 0; //pmpaddr0
                            //0x3a0: rval = 0; //pmpcfg0
                            //0xf12: rval = 0x00000000; //marchid
                            //0xf13: rval = 0x00000000; //mimpid
                            //0xf14: rval = 0x00000000; //mhartid
                            _default => {
                                handler.othercsr_read(csrno, rval);
                            }
                        }

                        match microop {
                            0b001 => {
                                writeval = rs1;
                            } //CSRRW
                            0b010 => {
                                writeval = rval | rs1;
                            } //CSRRS
                            0b011 => {
                                writeval = rval & !rs1;
                            } //CSRRC
                            0b101 => {
                                writeval = rs1imm;
                            } //CSRRWI
                            0b110 => {
                                writeval = rval | rs1imm;
                            } //CSRRSI
                            0b111 => {
                                writeval = rval & !rs1imm;
                            } //CSRRCI
                            _default => {}
                        }

                        match csrno {
                            0x340 => state.mscratch = writeval,
                            0x305 => state.mtvec = writeval,
                            0x304 => state.mie = writeval,
                            0x344 => state.mip = writeval,
                            0x341 => state.mepc = writeval,
                            0x300 => state.mstatus = writeval, //mstatus
                            0x342 => state.mcause = writeval,
                            0x343 => state.mtval = writeval,
                            //0x3a0:  //pmpcfg0
                            //0x3B0:  //pmpaddr0
                            //0xf11:  //mvendorid
                            //0xf12:  //marchid
                            //0xf13:  //mimpid
                            //0xf14:  //mhartid
                            //0x301:  //misa
                            _default => {
                                handler.othercsr_write(image, csrno, writeval);
                            }
                        }
                    } else if microop == 0b000 {
                        // "SYSTEM"

                        rdid = 0;
                        if csrno == 0x105 {
                            //WFI (Wait for interrupts)

                            state.mstatus |= 8; //Enable interrupts
                            state.extraflags |= 4; //Infor environment we want to go to sleep.
                            state.pc = pc + 4;
                            return Ok(1);
                        } else if (csrno & 0xff) == 0x02 {
                            // MRET

                            //https://raw.githubusercontent.com/riscv/virtual-memory/main/specs/663-Svpbmt.pdf
                            //Table 7.6. MRET then in mstatus/mstatush sets MPV=0, MPP=0, MIE=MPIE, and MPIE=1. La
                            // Should also update mstatus to reflect correct mode.
                            let startmstatus = state.mstatus;
                            let startextraflags = state.extraflags;
                            let newstatus =
                                ((startmstatus & 0x80) >> 4) | ((startextraflags & 3) << 11) | 0x80;
                            state.mstatus = newstatus;
                            let newflags = (startextraflags & !3) | ((startmstatus >> 11) & 3);
                            state.extraflags = newflags;
                            pc = state.mepc - 4;
                        } else {
                            match csrno {
                                0 => {
                                    trap = if (state.extraflags & 3) != 0 {
                                        11 + 1
                                    } else {
                                        8 + 1
                                    };
                                } // ECALL; 8 = "Environment call from U-mode"; 11 = "Environment call from M-mode"
                                1 => {
                                    trap = 3 + 1;
                                } // EBREAK 3 = "Breakpoint"
                                _default => {
                                    trap = 2 + 1;
                                } // Illegal opcode.
                            }
                        }
                    } else {
                        trap = 2 + 1;
                    } // Note micrrop 0b100 == undefined.
                }
                0b0101111 => {
                    // RV32A

                    let mut rs1 = state.reg((ir >> 15) & 0x1f);
                    let mut rs2 = state.reg((ir >> 20) & 0x1f);
                    let irmid = (ir >> 27) & 0x1f;

                    rs1 -= MINIRV32_RAM_IMAGE_OFFSET;

                    // We don't implement load/store from UART or CLNT with RV32A here.

                    if rs1 >= MINI_RV32_RAM_SIZE - 3 {
                        trap = 7 + 1; //Store/AMO access fault
                        rval = rs1 + MINIRV32_RAM_IMAGE_OFFSET;
                    } else {
                        rval = image.load32(rs1);

                        // Referenced a little bit of https://github.com/franzflasch/riscv_em/blob/master/src/core/core.c
                        let mut dowrite = true;
                        match irmid {
                            0b00010 => {
                                dowrite = false;
                            } //LR.W
                            0b00011 => {
                                rval = 0;
                            } //SC.W (Lie and always say it's good)
                            0b00001 => {} //AMOSWAP.W
                            0b00000 => {
                                rs2 = rs2.wrapping_add(rval);
                            } //AMOADD.W
                            0b00100 => {
                                rs2 ^= rval;
                            } //AMOXOR.W
                            0b01100 => {
                                rs2 &= rval;
                            } //AMOAND.W
                            0b01000 => {
                                rs2 |= rval;
                            } //AMOOR.W
                            0b10000 => {
                                rs2 = if (rs2 as i32) < (rval as i32) {
                                    rs2
                                } else {
                                    rval
                                }
                            } //AMOMIN.W
                            0b10100 => {
                                rs2 = if (rs2 as i32) > (rval as i32) {
                                    rs2
                                } else {
                                    rval
                                }
                            } //AMOMAX.W
                            0b11000 => {
                                rs2 = if rs2 < rval { rs2 } else { rval };
                            } //AMOMINU.W
                            0b11100 => {
                                rs2 = if rs2 > rval { rs2 } else { rval };
                            } //AMOMAXU.W
                            _default => {
                                trap = 2 + 1;
                                dowrite = false;
                            } //Not supported.
                        }
                        if dowrite {
                            image.store32(rs1, rs2);
                        }
                    }
                }
                _default => {
                    trap = 2 + 1; // Fault: Invalid opcode.
                }
            }

            if trap == 0 {
                if rdid != 0 {
                    state.regset(rdid, rval);
                }
                // Write back register.
                else if (state.mip & (1 << 7) != 0)
                    && (state.mie & (1 << 7) != 0/*mtie*/)
                    && (state.mstatus & 0x8 != 0/*mie*/)
                {
                    trap = 0x80000007; // Timer interrupt.
                }
            }
        }

        handler.postexec(pc, ir, &mut trap)?;

        // Handle traps and interrupts.
        if trap != 0 {
            if trap & 0x80000000 != 0 {
                // If prefixed with 0x100, it's an interrupt, not a trap.

                state.mcause = trap;
                state.mtval = 0;
                pc += 4; // PC needs to point to where the PC will return to.
            } else {
                state.mcause = trap - 1;
                state.mtval = if trap > 5 && trap <= 8 { rval } else { pc };
            }
            state.mepc = pc; //TRICKY: The kernel advances mepc automatically.
                             //CSR( mstatus ) & 8 = MIE, & 0x80 = MPIE
                             // On an interrupt, the system moves current MIE into MPIE
            let newmstatus = ((state.mstatus & 0x08) << 4) | ((state.extraflags & 3) << 11);
            state.mstatus = newmstatus;
            pc = state.mtvec.wrapping_sub(4);

            // XXX TODO: Do we actually want to check here? Is this correct?
            if (trap & 0x80000000) == 0 {
                state.extraflags |= 3;
            }
        }

        state.pc = pc.wrapping_add(4);
    }
    return Ok(0);
}

static DEFAULT64MBDTB: &[u8] = &[
    0xd0, 0x0d, 0xfe, 0xed, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x05, 0x00,
    0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xe3, 0x00, 0x00, 0x04, 0xc8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1b, 0x72, 0x69, 0x73, 0x63,
    0x76, 0x2d, 0x6d, 0x69, 0x6e, 0x69, 0x6d, 0x61, 0x6c, 0x2d, 0x6e, 0x6f, 0x6d, 0x6d, 0x75, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x26, 0x72, 0x69, 0x73, 0x63,
    0x76, 0x2d, 0x6d, 0x69, 0x6e, 0x69, 0x6d, 0x61, 0x6c, 0x2d, 0x6e, 0x6f, 0x6d, 0x6d, 0x75, 0x2c,
    0x71, 0x65, 0x6d, 0x75, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x63, 0x68, 0x6f, 0x73,
    0x65, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x2c,
    0x65, 0x61, 0x72, 0x6c, 0x79, 0x63, 0x6f, 0x6e, 0x3d, 0x75, 0x61, 0x72, 0x74, 0x38, 0x32, 0x35,
    0x30, 0x2c, 0x6d, 0x6d, 0x69, 0x6f, 0x2c, 0x30, 0x78, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x2c, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x20, 0x63, 0x6f, 0x6e, 0x73, 0x6f, 0x6c,
    0x65, 0x3d, 0x74, 0x74, 0x79, 0x53, 0x30, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x40, 0x38, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x35, 0x6d, 0x65, 0x6d, 0x6f,
    0x72, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x41,
    0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xff, 0xc0, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x63, 0x70, 0x75, 0x73, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x45, 0x00, 0x0f, 0x42, 0x40,
    0x00, 0x00, 0x00, 0x01, 0x63, 0x70, 0x75, 0x40, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x35, 0x63, 0x70, 0x75, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x60, 0x6f, 0x6b, 0x61, 0x79, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x1b, 0x72, 0x69, 0x73, 0x63,
    0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x67,
    0x72, 0x76, 0x33, 0x32, 0x69, 0x6d, 0x61, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b,
    0x00, 0x00, 0x00, 0x71, 0x72, 0x69, 0x73, 0x63, 0x76, 0x2c, 0x6e, 0x6f, 0x6e, 0x65, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x72, 0x75, 0x70, 0x74, 0x2d, 0x63, 0x6f,
    0x6e, 0x74, 0x72, 0x6f, 0x6c, 0x6c, 0x65, 0x72, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x7a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0f,
    0x00, 0x00, 0x00, 0x1b, 0x72, 0x69, 0x73, 0x63, 0x76, 0x2c, 0x63, 0x70, 0x75, 0x2d, 0x69, 0x6e,
    0x74, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x58,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x63, 0x70, 0x75, 0x2d, 0x6d, 0x61, 0x70, 0x00, 0x00, 0x00, 0x00, 0x01, 0x63, 0x6c, 0x75, 0x73,
    0x74, 0x65, 0x72, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x63, 0x6f, 0x72, 0x65,
    0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xa0,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x73, 0x6f, 0x63, 0x00, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x1b, 0x73, 0x69, 0x6d, 0x70, 0x6c, 0x65, 0x2d, 0x62,
    0x75, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa4,
    0x00, 0x00, 0x00, 0x01, 0x75, 0x61, 0x72, 0x74, 0x40, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
    0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xab,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x41,
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1b, 0x6e, 0x73, 0x31, 0x36,
    0x38, 0x35, 0x30, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x70, 0x6f, 0x77, 0x65,
    0x72, 0x6f, 0x66, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0xbb, 0x00, 0x00, 0x55, 0x55, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0xc1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10,
    0x00, 0x00, 0x00, 0x1b, 0x73, 0x79, 0x73, 0x63, 0x6f, 0x6e, 0x2d, 0x70, 0x6f, 0x77, 0x65, 0x72,
    0x6f, 0x66, 0x66, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x72, 0x65, 0x62, 0x6f,
    0x6f, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xbb,
    0x00, 0x00, 0x77, 0x77, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xc1,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xc8,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x1b,
    0x73, 0x79, 0x73, 0x63, 0x6f, 0x6e, 0x2d, 0x72, 0x65, 0x62, 0x6f, 0x6f, 0x74, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x73, 0x79, 0x73, 0x63, 0x6f, 0x6e, 0x40, 0x31,
    0x31, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10,
    0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x11, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x1b,
    0x73, 0x79, 0x73, 0x63, 0x6f, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x63, 0x6c, 0x69, 0x6e, 0x74, 0x40, 0x31, 0x31, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xcf, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x1b,
    0x00, 0x00, 0x00, 0x1b, 0x73, 0x69, 0x66, 0x69, 0x76, 0x65, 0x2c, 0x63, 0x6c, 0x69, 0x6e, 0x74,
    0x30, 0x00, 0x72, 0x69, 0x73, 0x63, 0x76, 0x2c, 0x63, 0x6c, 0x69, 0x6e, 0x74, 0x30, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x09,
    0x23, 0x61, 0x64, 0x64, 0x72, 0x65, 0x73, 0x73, 0x2d, 0x63, 0x65, 0x6c, 0x6c, 0x73, 0x00, 0x23,
    0x73, 0x69, 0x7a, 0x65, 0x2d, 0x63, 0x65, 0x6c, 0x6c, 0x73, 0x00, 0x63, 0x6f, 0x6d, 0x70, 0x61,
    0x74, 0x69, 0x62, 0x6c, 0x65, 0x00, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x00, 0x62, 0x6f, 0x6f, 0x74,
    0x61, 0x72, 0x67, 0x73, 0x00, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x74, 0x79, 0x70, 0x65,
    0x00, 0x72, 0x65, 0x67, 0x00, 0x74, 0x69, 0x6d, 0x65, 0x62, 0x61, 0x73, 0x65, 0x2d, 0x66, 0x72,
    0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x79, 0x00, 0x70, 0x68, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x00,
    0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x00, 0x72, 0x69, 0x73, 0x63, 0x76, 0x2c, 0x69, 0x73, 0x61,
    0x00, 0x6d, 0x6d, 0x75, 0x2d, 0x74, 0x79, 0x70, 0x65, 0x00, 0x23, 0x69, 0x6e, 0x74, 0x65, 0x72,
    0x72, 0x75, 0x70, 0x74, 0x2d, 0x63, 0x65, 0x6c, 0x6c, 0x73, 0x00, 0x69, 0x6e, 0x74, 0x65, 0x72,
    0x72, 0x75, 0x70, 0x74, 0x2d, 0x63, 0x6f, 0x6e, 0x74, 0x72, 0x6f, 0x6c, 0x6c, 0x65, 0x72, 0x00,
    0x63, 0x70, 0x75, 0x00, 0x72, 0x61, 0x6e, 0x67, 0x65, 0x73, 0x00, 0x63, 0x6c, 0x6f, 0x63, 0x6b,
    0x2d, 0x66, 0x72, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x79, 0x00, 0x76, 0x61, 0x6c, 0x75, 0x65,
    0x00, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x00, 0x72, 0x65, 0x67, 0x6d, 0x61, 0x70, 0x00, 0x69,
    0x6e, 0x74, 0x65, 0x72, 0x72, 0x75, 0x70, 0x74, 0x73, 0x2d, 0x65, 0x78, 0x74, 0x65, 0x6e, 0x64,
    0x65, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
