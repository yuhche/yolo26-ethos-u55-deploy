# Firmware Model Assets

Place the Vela-compiled model here:

```text
model_int8_vela.tflite
```

Generate a C array for firmware builds:

```bash
xxd -i model_int8_vela.tflite > model_data.cc
```

The generated symbols must match `firmware/src/main.cc`:

```c
extern const unsigned char model_int8_vela_tflite[];
extern const unsigned int model_int8_vela_tflite_len;
```

If `xxd` generates different symbol names, either rename the file before running `xxd` or update `main.cc`.
