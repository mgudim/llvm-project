# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vredsum.vs v8, v4, v20, v0.t
# CHECK-INST: vredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 004a2457 <unknown>

vredsum.vs v8, v4, v20
# CHECK-INST: vredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 024a2457 <unknown>

vredmaxu.vs v8, v4, v20, v0.t
# CHECK-INST: vredmaxu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 184a2457 <unknown>

vredmaxu.vs v8, v4, v20
# CHECK-INST: vredmaxu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1a4a2457 <unknown>

vredmax.vs v8, v4, v20, v0.t
# CHECK-INST: vredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1c4a2457 <unknown>

vredmax.vs v8, v4, v20
# CHECK-INST: vredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1e4a2457 <unknown>

vredminu.vs v8, v4, v20, v0.t
# CHECK-INST: vredminu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 104a2457 <unknown>

vredminu.vs v8, v4, v20
# CHECK-INST: vredminu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 124a2457 <unknown>

vredmin.vs v8, v4, v20, v0.t
# CHECK-INST: vredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x14]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 144a2457 <unknown>

vredmin.vs v8, v4, v20
# CHECK-INST: vredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x16]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 164a2457 <unknown>

vredand.vs v8, v4, v20, v0.t
# CHECK-INST: vredand.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 044a2457 <unknown>

vredand.vs v8, v4, v20
# CHECK-INST: vredand.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 064a2457 <unknown>

vredor.vs v8, v4, v20, v0.t
# CHECK-INST: vredor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 084a2457 <unknown>

vredor.vs v8, v4, v20
# CHECK-INST: vredor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0a4a2457 <unknown>

vredxor.vs v8, v4, v20, v0.t
# CHECK-INST: vredxor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c4a2457 <unknown>

vredxor.vs v8, v4, v20
# CHECK-INST: vredxor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e4a2457 <unknown>

vwredsumu.vs v8, v4, v20, v0.t
# CHECK-INST: vwredsumu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c04a0457 <unknown>

vwredsumu.vs v8, v4, v20
# CHECK-INST: vwredsumu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c24a0457 <unknown>

vwredsum.vs v8, v4, v20, v0.t
# CHECK-INST: vwredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c44a0457 <unknown>

vwredsum.vs v8, v4, v20
# CHECK-INST: vwredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c64a0457 <unknown>

vredsum.vs v0, v4, v20, v0.t
# CHECK-INST: vredsum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x20,0x4a,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 004a2057 <unknown>
