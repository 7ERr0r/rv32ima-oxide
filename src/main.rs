// Copyright 2022 Charles Lohr, you may use this file or any portions herein under any of the BSD, MIT, or CC0 licenses.

/**
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



pub fn main() {
    let ram_amt: u32 = 64 * 1024 * 1024;
    let mut instct: u64 = -1 as i32 as u32 as u64;
    let show_help = 0;
    let time_divisor = 1;
    let fixed_update = false;
    let do_sleep = true;
    let single_step = false;

    let ram_image: Vec<u8>;
    let mut core: MiniRV32IMAState;

    let dtb_ptr: u32 = 0;
    ram_image = Vec::with_capacity(ram_amt as usize);

    let mut ram_image = RVImage { image: ram_image };

    // The core lives at the end of RAM.
    //core = (MiniRV32IMAState *)(ram_image + ram_amt - sizeof( struct MiniRV32IMAState ));
    core = MiniRV32IMAState::default();
    core.pc = MINIRV32_RAM_IMAGE_OFFSET;
    core.regs[10] = 0x00; //hart ID
    core.regs[11] = if dtb_ptr != 0 {
        dtb_ptr + MINIRV32_RAM_IMAGE_OFFSET
    } else {
        0
    }; //dtb_pa (Must be valid pointer) (Should be pointer to dtb)
    core.extraflags |= 3; // Machine-mode.

    // Image is loaded.

    let mut last_time: u64 = if fixed_update {
        0
    } else {
        time_now_micros() / time_divisor
    };
    let instrs_per_flip = if single_step { 1 } else { 1024 };
    let mut rt: u64 = 0;
    while rt < instct + 1 {
        //u64 * this_ccount = ((u64*)&core->cyclel);
        let elapsed_us: u64;
        if fixed_update {
            elapsed_us = core.cycle() / time_divisor - last_time;
        } else {
            elapsed_us = time_now_micros() / time_divisor - last_time;
        }
        last_time += elapsed_us;

        if single_step {
            dump_state(&core, &ram_image, &ram_amt);
        }
        let ret = mini_rv32_ima_step(
            &mut core,
            &mut ram_image,
            0,
            elapsed_us as u32,
            instrs_per_flip,
        ); // Execute upto 1024 cycles before breaking out.
        match ret {
            0 => break,
            1 => {
                if do_sleep {
                    //MiniSleep();
                    core.set_cycle(core.cycle() + instrs_per_flip as u64);
                }
            }
            3 => {
                instct = 0;
            }
            0x7777 => {
                //goto restart;	//syscon code for restart
            }
            0x5555 => {
                println!("POWEROFF@{}{}", core.cycleh, core.cyclel);
                return; //syscon code for power-off
            }
            _default => {
                println!("Unknown failure");
                break;
            }
        }

        rt += instrs_per_flip as u64;
    }

    dump_state(&core, &ram_image, &ram_amt);
}

pub fn dump_state(core: &MiniRV32IMAState, image: &RVImage, ram_amt: &u32) {
    let pc = core.pc;
    let pc_offset = pc - MINIRV32_RAM_IMAGE_OFFSET;
    let mut ir = 0;

    println!("PC: {:#08x} ", pc);
    if pc_offset < ram_amt - 3 {
        ir = image.load32(pc_offset);
        println!("[0x{:#08x}] ", ir);
    } else {
        println!("[xxxxxxxxxx] ");
    }
    let regs = &core.regs;
    println!( "Z:{:#08x} ra:{:#08x} sp:{:#08x} gp:{:#08x} tp:{:#08x} t0:{:#08x} t1:{:#08x} t2:{:#08x} s0:{:#08x} s1:{:#08x} a0:{:#08x} a1:{:#08x} a2:{:#08x} a3:{:#08x} a4:{:#08x} a5:{:#08x} ",
		regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7],
		regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15] );
    println!( "a6:{:#08x} a7:{:#08x} s2:{:#08x} s3:{:#08x} s4:{:#08x} s5:{:#08x} s6:{:#08x} s7:{:#08x} s8:{:#08x} s9:{:#08x} s10:{:#08x} s11:{:#08x} t3:{:#08x} t4:{:#08x} t5:{:#08x} t6:{:#08x}\n",
		regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23],
		regs[24], regs[25], regs[26], regs[27], regs[28], regs[29], regs[30], regs[31] );
}

pub fn time_now_micros() -> u64 {
    let start = std::time::SystemTime::now();
    let since_the_epoch = start
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards");
    since_the_epoch.as_micros() as u64
}

static MINI_RV32_RAM_SIZE: u32 = 1024 * 1024 * 16;
static MINIRV32_RAM_IMAGE_OFFSET: u32 = 0x80000000;

fn minirv32_postexec(_pc: u32, _ir: u32, _trap: u32) {}

fn minirv32_handle_mem_store_control(_addy: u32, _rs2: u32) {}

fn minirv32_handle_mem_load_control(_rsval: u32, _rval: u32) {}

fn minirv32_othercsr_write(_csrno: u32, _writeval: u32) {}

fn minirv32_othercsr_read(_csrno: u32, _rval: u32) {}

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

pub fn mini_rv32_ima_step(
    state: &mut MiniRV32IMAState,
    image: &mut RVImage,
    _v_proc_address: u32,
    elapsed_us: u32,
    count: i32,
) -> i32 {
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
        return 1;
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
        let ofs_pc: u32 = pc - MINIRV32_RAM_IMAGE_OFFSET;

        if ofs_pc >= MINI_RV32_RAM_SIZE {
            trap = 1 + 1; // Handle access violation on instruction read.
        } else if ofs_pc & 3 != 0 {
            trap = 1 + 0; //Handle PC-misaligned access
        } else {
            ir = image.load32(ofs_pc);
            let mut rdid: u32 = (ir >> 7) & 0x1f;

            match ir & 0x7f {
                0b0110111 => {
                    // LUI
                    rval = ir & 0xfffff000;
                }
                0b0010111 => {
                    // AUIPC
                    rval = pc + (ir & 0xfffff000);
                }
                0b1101111 => {
                    // JAL

                    let mut reladdy: u32 = ((ir & 0x80000000) >> 11)
                        | ((ir & 0x7fe00000) >> 20)
                        | ((ir & 0x00100000) >> 9)
                        | (ir & 0x000ff000);
                    if reladdy & 0x00100000 != 0 {
                        reladdy |= 0xffe00000; // Sign extension.
                    }
                    rval = pc + 4;
                    pc = pc + reladdy as u32 - 4;
                }
                0b1100111 => {
                    // JALR

                    let imm: u32 = ir >> 20;
                    let xx = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm_se: u32 = imm | xx;
                    rval = pc + 4;
                    pc = ((state.reg((ir >> 15) & 0x1f) + imm_se) & !1) - 4;
                }
                0b1100011 => {
                    // Branch

                    let mut immm4: u32 = ((ir & 0xf00) >> 7)
                        | ((ir & 0x7e000000) >> 20)
                        | ((ir & 0x80) << 4)
                        | ((ir >> 31) << 12);
                    if immm4 & 0x1000 != 0 {
                        immm4 |= 0xffffe000;
                    }
                    let rs1: u32 = state.reg((ir >> 15) & 0x1f);
                    let rs2: u32 = state.reg((ir >> 20) & 0x1f);
                    immm4 = pc + immm4 - 4;
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
                            if rs1 >= rs2 {
                                pc = immm4;
                            }
                        } //BGE
                        0b110 => {
                            if (rs1 as u32) < rs2 as u32 {
                                pc = immm4;
                            }
                        } //BLTU
                        0b111 => {
                            if (rs1 as u32) >= rs2 as u32 {
                                pc = immm4;
                            }
                        } //BGEU
                        _default => {
                            trap = 2 + 1;
                        }
                    }
                }
                0b0000011 => {
                    // Load

                    let rs1: u32 = state.reg((ir >> 15) & 0x1f);
                    let imm: u32 = ir >> 20;
                    let xx = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm_se: u32 = imm | xx;
                    let mut rsval: u32 = rs1 + imm_se;

                    rsval -= MINIRV32_RAM_IMAGE_OFFSET;
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
                                minirv32_handle_mem_load_control(rsval, rval);
                            }
                        } else {
                            trap = 5 + 1;
                            rval = rsval;
                        }
                    } else {
                        match (ir >> 12) & 0x7 {
                            //LB, LH, LW, LBU, LHU
                            0b000 => {
                                rval = image.load8(rsval) as i8 as u32;
                            }
                            0b001 => {
                                rval = image.load16(rsval) as i16 as u32;
                            }
                            0b010 => {
                                rval = image.load32(rsval);
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

                    let rs1: u32 = state.reg((ir >> 15) & 0x1f);
                    let rs2: u32 = state.reg((ir >> 20) & 0x1f);
                    let mut addy: u32 = ((ir >> 7) & 0x1f) | ((ir & 0xfe000000) >> 20);
                    if addy & 0x800 != 0 {
                        addy |= 0xfffff000;
                    }
                    addy += rs1 - MINIRV32_RAM_IMAGE_OFFSET;
                    rdid = 0;

                    if addy >= MINI_RV32_RAM_SIZE - 3 {
                        addy -= MINIRV32_RAM_IMAGE_OFFSET;
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
                                return rs2 as _; // NOTE: PC will be PC of Syscon.
                            } else {
                                minirv32_handle_mem_store_control(addy, rs2);
                            }
                        } else {
                            trap = 7 + 1; // Store access fault.
                            rval = addy + MINIRV32_RAM_IMAGE_OFFSET;
                        }
                    } else {
                        match (ir >> 12) & 0x7 {
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
                    let xx = if imm & 0x800 != 0 { 0xfffff000 } else { 0 };
                    let imm = imm | xx;
                    let rs1 = state.reg((ir >> 15) & 0x1f);
                    let is_reg = ir & 0b100000 != 0;
                    let rs2 = if is_reg { state.reg(imm & 0x1f) } else { imm };

                    if is_reg && (ir & 0x02000000 != 0) {
                        match (ir >> 12) & 7 {
                            //0x02000000 = RV32M
                            0b000 => {
                                rval = rs1 * rs2;
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
                        match (ir >> 12) & 7 {
                            // These could be either op-immediate or op commands.  Be careful.
                            0b000 => {
                                rval = if is_reg && (ir & 0x40000000) != 0 {
                                    rs1 - rs2
                                } else {
                                    rs1 + rs2
                                };
                            }
                            0b001 => {
                                rval = rs1 << rs2;
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
                                    ((rs1 as i32) >> rs2) as u32
                                } else {
                                    rs1 >> rs2
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
                                minirv32_othercsr_read(csrno, rval);
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
                                minirv32_othercsr_write(csrno, writeval);
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
                            return 1;
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
                                rs2 += rval;
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

        minirv32_postexec(pc, ir, trap);

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
            pc = state.mtvec - 4;

            // XXX TODO: Do we actually want to check here? Is this correct?
            if (trap & 0x80000000) == 0 {
                state.extraflags |= 3;
            }
        }

        state.pc = pc + 4;
    }
    return 0;
}
