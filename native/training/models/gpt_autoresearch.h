// gpt_autoresearch.h — autoresearch GPT config for ANE training
// Matches train_mac.py GPTConfig defaults for cross-validation
#pragma once

#define MODEL_NAME "GPT-autoresearch"

#define DIM 768
#define HIDDEN 2048       // 4 * DIM (ReluSquared FFN, no gate)
#define HEADS 6
#define KV_HEADS 6
#define HD (DIM/HEADS)    // = 128
#define GQA_RATIO 1       // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 768 = DIM
#define KV_DIM (KV_HEADS * HD) // = 768 = DIM
#define SEQ 256           // Start small for ANE testing, scale up
#define NLAYERS 8         // Default depth
#define VOCAB 32768       // autoresearch vocab size

#define CKPT_PATH "ane_autoresearch_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "data/tinystories_data00.bin"
