rv32ima-oxide
===
Copy-paste from https://github.com/cnlohr/mini-rv32ima into Rust
```
Options:
  -m, --memory-ram <MEMORY_RAM>
          RAM amount [default: 64 * 1024 * 1024]
  -c, --count-instruction <COUNT_INSTRUCTION>
          Instruction count
  -r, --repeat <REPEAT>
          Number of times to repeat - useful for benchmarking [default: 1]
  -l, --lock-time-update
          Lock time base to instruction count. Use locked time update to disable realtime system clock
  -t, --time-divisor <TIME_DIVISOR>
          Slowdown used for a fixed-time step [default: 1]
  -s, --single-step
          Useful for debugging. Prints processor state in every step
  -p, --presto-disable-sleep
          Use to disable micro-sleep when wfi
  -f, --file-image <FILE_IMAGE>
          Path to running image file
```


# Run


```shell
./rv32ima-oxide$ cargo run
   Compiling rv32ima-oxide v0.1.0 (/home/ubuntu/Documents/rsws/rv32ima-oxide)
    Finished dev [unoptimized + debuginfo] target(s) in 0.26s
     Running `target/debug/rv32ima-oxide`
uart: "\0[    0.000000] Linux version 5.18.0 (cnlohr@cnlohr-1520) (riscv32-buildroot-linux-uclibc-gcc.br_real (Buildroot -g91b88fa1) 10.3.0, GNU ld (GNU Binutils) 2.37) #7 Sun Nov 27 07:07:08 EST 2022\r\n"
uart: "[    0.000000] Machine model: riscv-minimal-nommu,qemu\r\n"
uart: "[    0.000000] earlycon: uart8250 at MMIO 0x10000000 (options '1000000')\r\n"
uart: "[    0.000000] printk: bootconsole [uart8250] enabled\r\n"
uart: "[    0.000000] Zone ranges:\r\n"
uart: "[    0.000000]   Normal   [mem 0x0000000080000000-0x0000000083ffefff]\r\n"
uart: "[    0.000000] Movable zone start for each node\r\n"
uart: "[    0.000000] Early memory node ranges\r\n"
uart: "[    0.000000]   node   0: [mem 0x0000000080000000-0x0000000083ffefff]\r\n"
uart: "[    0.000000] Initmem setup node 0 [mem 0x0000000080000000-0x0000000083ffefff]\r\n"
uart: "[    0.000000] riscv: base ISA extensions aim\r\n"
uart: "[    0.000000] riscv: ELF capabilities aim\r\n"
uart: "[    0.000000] Built 1 zonelists, mobility grouping on.  Total pages: 16255\r\n"
uart: "[    0.000000] Kernel command line: earlycon=uart8250,mmio,0x10000000,1000000 console=ttyS0\r\n"
uart: "[    0.000000] Dentry cache hash table entries: 8192 (order: 3, 32768 bytes, linear)\r\n"
uart: "[    0.000000] Inode-cache hash table entries: 4096 (order: 2, 16384 bytes, linear)\r\n"
uart: "[    0.000000] Sorting __ex_table...\r\n"
uart: "[    0.000000] mem auto-init: stack:off, heap alloc:off, heap free:off\r\n"
uart: "[    0.000000] Memory: 61936K/65532K available (1346K kernel code, 271K rwdata, 149K rodata, 1105K init, 108K bss, 3596K reserved, 0K cma-reserved)\r\n"
uart: "[    0.000000] NR_IRQS: 64, nr_irqs: 64, preallocated irqs: 0\r\n"
uart: "[    0.000000] riscv-intc: 32 local interrupts mapped\r\n"
uart: "[    0.000000] clint: clint@11000000: timer running at 1000000 Hz\r\n"
uart: "[    0.000000] clocksource: clint_clocksource: mask: 0xffffffffffffffff max_cycles: 0x1d854df40, max_idle_ns: 3526361616960 ns\r\n"
uart: "[    0.000000] sched_clock: 64 bits at 1000kHz, resolution 1000ns, wraps every 2199023255500ns\r\n"
uart: "[    0.006157] Console: colour dummy device 80x25\r\n"
uart: "[    0.007538] Calibrating delay loop (skipped), value calculated using timer frequency.. 2.00 BogoMIPS (lpj=4000)\r\n"
uart: "[    0.009115] pid_max: default: 4096 minimum: 301\r\n"
uart: "[    0.012291] Mount-cache hash table entries: 1024 (order: 0, 4096 bytes, linear)\r\n"
uart: "[    0.013420] Mountpoint-cache hash table entries: 1024 (order: 0, 4096 bytes, linear)\r\n"
uart: "[    0.046820] devtmpfs: initialized\r\n"
uart: "[    0.112500] clocksource: jiffies: mask: 0xffffffff max_cycles: 0xffffffff, max_idle_ns: 7645041785100000 ns\r\n"
uart: "[    0.113907] futex hash table entries: 16 (order: -5, 192 bytes, linear)\r\n"
uart: "[    0.251762] clocksource: Switched to clocksource clint_clocksource\r\n"
uart: "[    0.630131] workingset: timestamp_bits=30 max_order=14 bucket_order=0\r\n"
uart: "[    1.130465] Serial: 8250/16550 driver, 1 ports, IRQ sharing disabled\r\n"
uart: "[    1.176870] printk: console [ttyS0] disabled\r\n"
uart: "[    1.177907] 10000000.uart: ttyS0 at MMIO 0x10000000 (irq = 0, base_baud = 1048576) is a XR16850\r\n"
uart: "m[    1.179145] printk: console [ttyS0] enabled\r\n"
uart: "[    1.179145] printk: console [ttyS0] enabled\r\n"
uart: "[    1.180060] printk: bootconsole [uart8250] disabled\r\n"
uart: "[    1.180060] printk: bootconsole [uart8250] disabled\r\n"
uart: "``m"
uart: "[    1.582658] Freeing unused kernel image (initmem) memory: 1104K\r\n"
uart: "[    1.583485] This architecture does not have kernel memory protection.\r\n"
uart: "[    1.584773] Run /init as init process\r\n"
uart: "\r\r\n"
uart: "Welcome to Buildroot\r\n"
uart: "\rbuildroot login: "
^C

```


Benchmarks
===
Best times in 20 rounds:

```
time bash -c "mini-rv32ima -f DownloadedImage -c 46000000 -p -l > out.txt"

real    0m0,305s
user    0m0,262s
sys     0m0,043s


time bash -c "rv32ima-oxide -c 46000000 -p -l > out.txt"

real    0m0,309s
user    0m0,294s
sys     0m0,013s
```