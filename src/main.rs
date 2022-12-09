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


	// #define MINIRV32WARN( x... );



	// #define MINIRV32_DECORATE static

static MINI_RV32_RAM_SIZE: u32 = 1024*1024*16;

	// #define MINIRV32_RAM_IMAGE_OFFSET  0x80000000
static MINIRV32_RAM_IMAGE_OFFSET: u32 = 0x80000000;


	// #define MINIRV32_POSTEXEC(...);

	fn MINIRV32_POSTEXEC(a: u32, b: u32, trap: u32) {

	}

	// #define MINIRV32_HANDLE_MEM_STORE_CONTROL(...);

	fn MINIRV32_HANDLE_MEM_STORE_CONTROL(a: u32, b: u32) {

	}

	// #define MINIRV32_HANDLE_MEM_LOAD_CONTROL(...);

fn MINIRV32_HANDLE_MEM_LOAD_CONTROL(a: u32, b: u32) {

}



	// #define MINIRV32_OTHERCSR_WRITE(...);
	fn MINIRV32_OTHERCSR_WRITE(a: u32, b: u32) {

	}


	// #define MINIRV32_OTHERCSR_READ(...);
	fn MINIRV32_OTHERCSR_READ(a: u32, b: u32) {

	}


	// #define MINIRV32_STORE4( ofs, val ) *(u32*)(image + ofs) = val
	// #define MINIRV32_STORE2( ofs, val ) *(u16*)(image + ofs) = val
	// #define MINIRV32_STORE1( ofs, val ) *(u8*)(image + ofs) = val
	// #define MINIRV32_LOAD4( ofs ) *(u32*)(image + ofs)
	// #define MINIRV32_LOAD2( ofs ) *(u16*)(image + ofs)
	// #define MINIRV32_LOAD1( ofs ) *(u8*)(image + ofs)


// As a note: We quouple-ify these, because in HLSL, we will be operating with
// uint4's.  We are going to uint4 data to/from system RAM.
//
// We're going to try to keep the full processor state to 12 x uint4.
struct MiniRV32IMAState { 
	pub regs: [u32; 32], //u32 regs[32];

	pub pc: u32, //u32 pc;
	pub mstatus: u32, // mstatus;
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
}

struct RVImage {
    image: Vec<u8>,

}
impl RVImage {
    pub fn load32(&self, offset: u32) -> u32 {
		let slice = &self.image[offset as usize..(offset+4) as usize];
        <u32>::from_le_bytes(unsafe {
            *(slice
                .as_ptr() as *const [u8; core::mem::size_of::<u32>()])
        })
    }
	pub fn load16(&self, offset: u32) -> u16 {
		let slice = &self.image[offset as usize..(offset+2) as usize];
        <u16>::from_le_bytes(unsafe {
            *(slice
                .as_ptr() as *const [u8; core::mem::size_of::<u16>()])
        })
    }
	pub fn load8(&self, offset: u32) -> u8 {
		self.image[offset as usize]
    }
	pub fn store32(&mut self, offset: u32, value: u32) {

	}
	pub fn store16(&mut self, offset: u32, value: u16) {


	}
	pub fn store8(&mut self, offset: u32, value: u8) {

	}
}


// #ifdef MINIRV32_IMPLEMENTATION

// #define CSR( x ) state->x
// #define SETCSR( x, val ) { state->x = val; }
// #define REG( x ) state->regs[x]
// #define REGSET( x, val ) { state->regs[x] = val; }

// macro_rules! csr {
//     // Base case:
//     ($x:expr) => (
//         state.x
//     );
//     // `$x` followed by at least one `$y,`
//     // ($x:expr, $($y:expr),+) => (
//     //     // Call `find_min!` on the tail `$y`
//     //     std::cmp::min($x, find_min!($($y),+))
//     // )
// }

pub fn MiniRV32IMAStep(state: &mut MiniRV32IMAState, image: RVImage, vProcAddress: u32, elapsedUs: u32, count: i32 ) -> i32 {
	let new_timer: u32 = state.timerl + elapsedUs;
	if new_timer < state.timerl {
        state.timerh += 1;
    }
	state.timerl = new_timer;

	// Handle Timer interrupt.
	if ( state.timerh > state.timermatchh
        || ( state.timerh == state.timermatchh && state.timerl > state.timermatchl ) ) 
        && ( state.timermatchh != 0 || state.timermatchl != 0 )
	{
		state.extraflags &= !4; // Clear WFI
		state.mip |= 1<<7; //MTIP of MIP // https://stackoverflow.com/a/61916199/2926815  Fire interrupt.
	}
	else {
		state.mip &= !(1<<7);
    }
	// If WFI, don't run processor.
	if (state.extraflags & 4) != 0 {
        return 1;
    }
    let icount: i32;

    for icount in 0..count {

		let ir = 0;
		let mut trap = 0; // If positive, is a trap or interrupt.  If negative, is fatal error.
		let rval = 0;

		// Increment both wall-clock and instruction count time.  (NOTE: Not strictly needed to run Linux)
		state.cyclel += 1;
		if state.cyclel == 0 {
            state.cycleh += 1;
        }

		let pc: u32 = state.pc;
		let ofs_pc: u32 = pc - MINIRV32_RAM_IMAGE_OFFSET;

		if ofs_pc  >= MINI_RV32_RAM_SIZE {
			trap = 1 + 1;  // Handle access violation on instruction read.
        }
		else if ofs_pc & 3 != 0 {
			trap = 1 + 0;  //Handle PC-misaligned access
        } else {
			ir = image.load32( ofs_pc );
			let rdid: u32 = (ir >> 7) & 0x1f;

			match ir & 0x7f {
				0b0110111 => { // LUI
					rval = ir & 0xfffff000;
				}
				0b0010111 => { // AUIPC
					rval = pc + ( ir & 0xfffff000 );
					
				}
				0b1101111 => { // JAL
				
					let reladdy: u32 = ((ir & 0x80000000)>>11) | ((ir & 0x7fe00000)>>20) | ((ir & 0x00100000)>>9) | ((ir&0x000ff000));
					if reladdy & 0x00100000 != 0 { 
						reladdy |= 0xffe00000; // Sign extension.
					}
					rval = pc + 4;
					pc = pc + reladdy as u32 - 4;
					
				}
				0b1100111 => { // JALR
				
					let imm: u32 = ir >> 20;
					let xx = if imm & 0x800 != 0 {0xfffff000}else{0};
					let imm_se: u32  = imm | xx;
					rval = pc + 4;
					pc = ( (state.reg( (ir >> 15) & 0x1f ) + imm_se) & !1) - 4;
					
				}
				0b1100011 => { // Branch
				
					let immm4: u32 = ((ir & 0xf00)>>7) | ((ir & 0x7e000000)>>20) | ((ir & 0x80) << 4) | ((ir >> 31)<<12);
					if immm4 & 0x1000 != 0 {immm4 |= 0xffffe000; }
					let rs1: u32  = state.reg((ir >> 15) & 0x1f);
					let rs2: u32  = state.reg((ir >> 20) & 0x1f);
					immm4 = pc + immm4 - 4;
					rdid = 0;
					match  ( ir >> 12 ) & 0x7 {
						// BEQ, BNE, BLT, BGE, BLTU, BGEU 
						0b000 => { if rs1 == rs2 { pc = immm4; }}
						0b001 => { if rs1 != rs2 { pc = immm4; }}
						0b100 => { if rs1 < rs2  {pc = immm4; }}
						0b101 => { if rs1 >= rs2 { pc = immm4; }} //BGE
						0b110 => { if (rs1 as u32) < rs2 as u32 { pc = immm4; }}   //BLTU
						0b111 => { if (rs1 as u32) >= rs2 as u32  {pc = immm4; }}  //BGEU
						_default => { trap = 2+1; }
					}
					break;
				}
				0b0000011 => { // Load
				
					let rs1: u32 = state.reg((ir >> 15) & 0x1f);
					let imm: u32 = ir >> 20;
					let xx = if imm & 0x800 != 0 {0xfffff000}else{0};
					let imm_se: u32 = imm | xx;
					let rsval: u32 = rs1 + imm_se;

					rsval -= MINIRV32_RAM_IMAGE_OFFSET;
					if rsval >= MINI_RV32_RAM_SIZE-3 {
						rsval -= MINIRV32_RAM_IMAGE_OFFSET;
						if( rsval >= 0x10000000 && rsval < 0x12000000 )  // UART, CLNT
						{
							if rsval == 0x1100bffc {// https://chromitem-soc.readthedocs.io/en/latest/clint.html
								rval = state.timerh;
							} else if rsval == 0x1100bff8 {
								rval = state.timerl;
							} else {
								MINIRV32_HANDLE_MEM_LOAD_CONTROL( rsval, rval );
							}
						}
						else
						{
							trap = 5+1;
							rval = rsval;
						}
					}
					else
					{
						match  ( ir >> 12 ) & 0x7 {
							//LB, LH, LW, LBU, LHU
							0b000 => { rval = image.load8( rsval ) as i8 as u32; }
							0b001 => { rval = image.load16( rsval ) as i16 as u32; }
							0b010 => { rval = image.load32( rsval ); }
							0b100 => { rval = image.load8( rsval ) as u32; }
							0b101 => { rval = image.load16( rsval ) as u32; }
							default => { trap = 2+1; }
						}
					}
				}
				0b0100011 => { // Store
				
					let rs1: u32 = state.reg((ir >> 15) & 0x1f);
					let rs2: u32 = state.reg((ir >> 20) & 0x1f);
					let addy: u32 = ( ( ir >> 7 ) & 0x1f ) | ( ( ir & 0xfe000000 ) >> 20 );
					if addy & 0x800 != 0 { addy |= 0xfffff000;}
					addy += rs1 - MINIRV32_RAM_IMAGE_OFFSET;
					rdid = 0;

					if( addy >= MINI_RV32_RAM_SIZE-3 )
					{
						addy -= MINIRV32_RAM_IMAGE_OFFSET;
						if( addy >= 0x10000000 && addy < 0x12000000 ) 
						{
							// Should be stuff like SYSCON, 8250, CLNT
							if( addy == 0x11004004 ) {//CLNT
							state.timermatchh = rs2;
							}else if( addy == 0x11004000 ){ //CLNT
								state.timermatchl = rs2;
							}else if( addy == 0x11100000 ) {//SYSCON (reboot, poweroff, etc.)
							
								state.pc = state.pc + 4;
								return rs2 as _; // NOTE: PC will be PC of Syscon.
							}else{
								MINIRV32_HANDLE_MEM_STORE_CONTROL( addy, rs2 );
							}
						}
						else
						{
							trap = (7+1); // Store access fault.
							rval = addy + MINIRV32_RAM_IMAGE_OFFSET;
						}
					}
					else
					{
						match ( ir >> 12 ) & 0x7 {
							//SB, SH, SW
							0b000 => { image.store8( addy, rs2 as u8 ); }
							0b001 => { image.store16( addy, rs2 as u16 ); }
							0b010 => { image.store32( addy, rs2 ); }
							_default => { trap = 2+1; }
						}
					}
					break;
				}
				
				0b0110011 | 0b0010011 => { // Op // Op-immediate
				
					let imm = ir >> 20;
					let xx = if imm & 0x800 != 0 {0xfffff000}else{0};
					let imm = imm | xx;
					let rs1 = state.reg((ir >> 15) & 0x1f);
					let is_reg = ir & 0b100000 != 0;
					let rs2 = if is_reg { state.reg(imm & 0x1f) }else{ imm};

					if is_reg && ( ir & 0x02000000 != 0) {
						match (ir>>12)&7 { //0x02000000 = RV32M
							0b000 => { rval = rs1 * rs2; } // MUL
							0b001 => { rval = (((rs1 as i32 as i64) * (rs2 as i32 as i64)) >> 32) as u32; } // MULH
							0b010 => { rval = (((rs1 as i32 as i64) * (rs2 as u64 as i64)) >> 32) as u32; } // MULHSU
							0b011 => { rval = (((rs1 as u64) * (rs2 as u64)) >> 32) as u32; } // MULHU
							0b100 => { if rs2 == 0 { rval = -1 as i32 as u32; }else{ rval = ((rs1 as i32) / (rs2 as i32)) as u32; }} // DIV
							0b101 => { if rs2 == 0 { rval = 0xffffffff; }else{ rval = rs1 / rs2; }}// DIVU
							0b110 => { if rs2 == 0 { rval = rs1; }else{ rval = ((rs1 as i32) % (rs2 as i32)) as u32; }} // REM
							0b111 => { if rs2 == 0 { rval = rs1; }else{ rval = rs1 % rs2; }} // REMU
						}
					}
					else
					{
						match (ir>>12)&7 { // These could be either op-immediate or op commands.  Be careful.
						
							0b000 => { rval = if is_reg && (ir & 0x40000000) != 0 {  rs1 - rs2 }else{ rs1 + rs2 }; }
							0b001 => { rval = rs1 << rs2; }
							0b010 => { rval = ((rs1 as i32) < (rs2 as i32)) as u32; }
							0b011 => { rval = (rs1 < rs2) as u32; }
							0b100 => { rval = rs1 ^ rs2; }
							0b101 => { rval = if ir & 0x40000000 != 0 {  ((rs1 as i32) >> rs2) as u32  }else{ rs1 >> rs2 }; }
							0b110 => { rval = rs1 | rs2; }
							0b111 => { rval = rs1 & rs2; }
						}
					}
					break;
				}
				0b0001111 => { 
					rdid = 0;   // fencetype = (ir >> 12) & 0b111; We ignore fences in this impl.
				}
				0b1110011 => { // Zifencei+Zicsr
				
					let csrno = ir >> 20;
					let microop = ( ir >> 12 ) & 0b111;
					if( (microop & 3) != 0 ) // It's a Zicsr function.
					{
						let rs1imm = (ir >> 15) & 0x1f;
						let rs1 = state.reg(rs1imm);
						let writeval = rs1;

						// https://raw.githubusercontent.com/riscv/virtual-memory/main/specs/663-Svpbmt.pdf
						// Generally, support for Zicsr
						match ( csrno ) {
						0x340 => { rval = state.mscratch  }
						0x305 => { rval = state.mtvec  }
						0x304 => { rval = state.mie  }
						0xC00 => { rval = state.cyclel  }
						0x344 => { rval = state.mip  }
						0x341 => { rval = state.mepc }
						0x300 => { rval = state.mstatus  } //mstatus
						0x342 => { rval = state.mcause }
						0x343 => { rval = state.mtval  }
						0xf11 => { rval = 0xff0ff0ff; } //mvendorid
						0x301 => { rval = 0x40401101; } //misa (XLEN=32, IMA+X)
						//0x3B0: rval = 0; break; //pmpaddr0
						//0x3a0: rval = 0; break; //pmpcfg0
						//0xf12: rval = 0x00000000; break; //marchid
						//0xf13: rval = 0x00000000; break; //mimpid
						//0xf14: rval = 0x00000000; break; //mhartid
						_default => {
							MINIRV32_OTHERCSR_READ( csrno, rval );
							
						}
						}	

						match( microop )
						{
							0b001 => { writeval = rs1; }  			//CSRRW
							0b010 => { writeval = rval | rs1; }		//CSRRS
							0b011 => { writeval = rval & !rs1; }		//CSRRC
							0b101 => { writeval = rs1imm; }			//CSRRWI
							0b110 => { writeval = rval | rs1imm; }	//CSRRSI
							0b111 => { writeval = rval & !rs1imm; }	//CSRRCI
						}

						match( csrno ) {
							0x340 => { state.mscratch = writeval }
							0x305 => { state.mtvec = writeval }
							0x304 => { state.mie = writeval }
							0x344 => { state.mip = writeval }
							0x341 => { state.mepc = writeval  }
							0x300 => { state.mstatus = writeval  } //mstatus
							0x342 => { state.mcause = writeval  }
							0x343 => { state.mtval = writeval  }
							//0x3a0: break; //pmpcfg0
							//0x3B0: break; //pmpaddr0
							//0xf11: break; //mvendorid
							//0xf12: break; //marchid
							//0xf13: break; //mimpid
							//0xf14: break; //mhartid
							//0x301: break; //misa
							_default => {
								MINIRV32_OTHERCSR_WRITE( csrno, writeval );
							}
						}
					}
					else if( microop == 0b000 ) // "SYSTEM"
					{
						rdid = 0;
						if( csrno == 0x105 ) //WFI (Wait for interrupts)
						{
							state.mstatus |= 8;    //Enable interrupts
							state.extraflags |= 4; //Infor environment we want to go to sleep.
							state.pc = pc + 4;
							return 1;
						}
						else if( ( ( csrno & 0xff ) == 0x02 ) )  // MRET
						{
							//https://raw.githubusercontent.com/riscv/virtual-memory/main/specs/663-Svpbmt.pdf
							//Table 7.6. MRET then in mstatus/mstatush sets MPV=0, MPP=0, MIE=MPIE, and MPIE=1. La
							// Should also update mstatus to reflect correct mode.
							let startmstatus = state.mstatus;
							let startextraflags = state.extraflags;
							let newstatus = (( startmstatus & 0x80) >> 4) | ((startextraflags&3) << 11) | 0x80;
							state.mstatus = newstatus;
							let newflags = (startextraflags & !3) | ((startmstatus >> 11) & 3);
							state.extraflags = newflags;
							pc = state.mepc -4;
						}
						else
						{
							match( csrno )
							{
							0 => { trap = if ( state.extraflags & 3) != 0 { (11+1) }else{ (8+1) }; } // ECALL; 8 = "Environment call from U-mode"; 11 = "Environment call from M-mode"
							1 => {	trap = (3+1); }// EBREAK 3 = "Breakpoint"
							_default => { trap = (2+1); } // Illegal opcode.
							}
						}
					}
					else{
						trap = (2+1);
					} 				// Note micrrop 0b100 == undefined.
					
				}
				0b0101111 => { // RV32A
				
					let rs1 = state.reg((ir >> 15) & 0x1f);
					let rs2 = state.reg((ir >> 20) & 0x1f);
					let irmid = ( ir>>27 ) & 0x1f;

					rs1 -= MINIRV32_RAM_IMAGE_OFFSET;

					// We don't implement load/store from UART or CLNT with RV32A here.

					if( rs1 >= MINI_RV32_RAM_SIZE-3 )
					{
						trap = (7+1); //Store/AMO access fault
						rval = rs1 + MINIRV32_RAM_IMAGE_OFFSET;
					}
					else
					{
						rval = image.load32( rs1 );

						// Referenced a little bit of https://github.com/franzflasch/riscv_em/blob/master/src/core/core.c
						let dowrite = true;
						match ( irmid )
						{
							0b00010 => {dowrite = false; } //LR.W
							0b00011 => {rval = 0; } //SC.W (Lie and always say it's good)
							0b00001 => {} //AMOSWAP.W
							0b00000 => {rs2 += rval; } //AMOADD.W
							0b00100 => {rs2 ^= rval; } //AMOXOR.W
							0b01100 => {rs2 &= rval; } //AMOAND.W
							0b01000 => {rs2 |= rval; } //AMOOR.W
							0b10000 => {rs2 = if((rs2 as i32)<(rval as i32)){rs2}else{rval} } //AMOMIN.W
							0b10100 => {rs2 = if((rs2 as i32)>(rval as i32)){rs2}else{rval} } //AMOMAX.W
							0b11000 => {rs2 = if(rs2<rval){rs2}else{rval}; } //AMOMINU.W
							0b11100 => {rs2 = if(rs2>rval){rs2}else{rval}; } //AMOMAXU.W
							_default => {trap = (2+1); dowrite = false; } //Not supported.
						}
						if( dowrite ) {
							image.store32( rs1, rs2 );
						}
					}
					break;
				}
				_default => {
					trap = (2+1); // Fault: Invalid opcode.
				}
			}

			if( trap == 0 )
			{
				if( rdid != 0 )
				{
					state.regset( rdid, rval );
				} // Write back register.
				else if( ( state.mip & (1<<7) != 0) && ( state.mie & (1<<7) != 0/*mtie*/ ) && ( state.mstatus & 0x8 != 0/*mie*/) )
				{
					trap = 0x80000007; // Timer interrupt.
				}
			}
		}

		MINIRV32_POSTEXEC( pc, ir, trap );

		// Handle traps and interrupts.
		if( trap != 0 )
		{
			if( trap & 0x80000000 != 0 ) // If prefixed with 0x100, it's an interrupt, not a trap.
			{
				state.mcause = trap;
				state.mtval = 0;
				pc += 4; // PC needs to point to where the PC will return to.
			}
			else
			{
				state.mcause = trap - 1;
				state.mtval = if (trap > 5 && trap <= 8) { rval }else{ pc };
			}
			state.mepc = pc; //TRICKY: The kernel advances mepc automatically.
			//CSR( mstatus ) & 8 = MIE, & 0x80 = MPIE
			// On an interrupt, the system moves current MIE into MPIE
			let newmstatus = (( state.mstatus & 0x08) << 4) | (( state.extraflags & 3 ) << 11);
			state.mstatus = newmstatus;
			pc = (state.mtvec - 4);

			// XXX TODO: Do we actually want to check here? Is this correct?
			if( (trap & 0x80000000) == 0){
				state.extraflags |= 3;
			}
		}

		state.pc = pc + 4;
	}
	return 0;
}
