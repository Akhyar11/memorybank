with open('train.py', 'r') as f:
    code = f.read()

code = code.replace("PREFETCH_QUEUE    = 8       # buffer batches di RAM sebelum GPU butuh\n", "PREFETCH_QUEUE    = 8       # buffer batches di RAM sebelum GPU butuh\nNUM_EPOCHS        = 2\n")

old_loop = """    # Pilih sumber data
    mode, data_path, tok_path = resolve_data_paths()
    if mode == 'npy':
        raw_gen = npy_epoch_generator(data_path, total_batch_size, SEQ_LEN)
    else:
        raw_gen = csv_epoch_generator(data_path, tok_path, total_batch_size, SEQ_LEN)

    dataloader = prefetch(raw_gen)   # background prefetch

    print("Starting Phase 1: Full Pre-Training (1 Epoch) ...\\n")
    start_time     = time.time()
    last_log_time  = start_time
    total_tokens   = 0
    accum_grads    = None

    for step, batch in enumerate(dataloader, 1):
        # Cast to int32 for JAX
        batch         = batch.astype(np.int32)
        sharded_batch = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))

        state, memory_state, metrics = train_step(state, memory_state, sharded_batch)
        total_tokens += total_batch_size * SEQ_LEN

        if step % LOG_INTERVAL == 0:
            now           = time.time()
            tok_per_sec   = (total_batch_size * SEQ_LEN * LOG_INTERVAL) / (now - last_log_time)
            elapsed       = int(now - start_time)
            ce_val        = float(unreplicate(metrics['ce_loss']))
            aux_val       = float(unreplicate(metrics['aux_loss']))
            expert_load   = np.array(unreplicate(metrics['expert_load']))

            # Format expert load: E0:12% E1:8% ...
            expert_str = ' '.join(
                f'E{i}:{v*100:.0f}%' for i, v in enumerate(expert_load)
            )
            print(
                f"Step {step:06d} | "
                f"CE {ce_val:.4f} | "
                f"Aux {aux_val:.4f} | "
                f"Speed {tok_per_sec:>8,.0f} tok/s | "
                f"Elapsed {elapsed//60}m {elapsed%60:02d}s"
            )
            print(f"          Expert load: [{expert_str}]")
            last_log_time = now"""

new_loop = """    print(f"Starting Phase 1: Full Pre-Training ({NUM_EPOCHS} Epochs) ...\\n")
    start_time     = time.time()
    last_log_time  = start_time
    total_tokens   = 0
    accum_grads    = None

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\\n========== EPOCH {epoch}/{NUM_EPOCHS} ==========")
        # Pilih sumber data
        mode, data_path, tok_path = resolve_data_paths()
        if mode == 'npy':
            raw_gen = npy_epoch_generator(data_path, total_batch_size, SEQ_LEN)
        else:
            raw_gen = csv_epoch_generator(data_path, tok_path, total_batch_size, SEQ_LEN)
    
        dataloader = prefetch(raw_gen)   # background prefetch
        
        for batch in dataloader:
            global_step += 1
            # Cast to int32 for JAX
            batch         = batch.astype(np.int32)
            sharded_batch = batch.reshape((num_devices, LOCAL_BATCH_SIZE, SEQ_LEN))
    
            state, memory_state, metrics = train_step(state, memory_state, sharded_batch)
            total_tokens += total_batch_size * SEQ_LEN
    
            if global_step % LOG_INTERVAL == 0:
                now           = time.time()
                tok_per_sec   = (total_batch_size * SEQ_LEN * LOG_INTERVAL) / (now - last_log_time)
                elapsed       = int(now - start_time)
                ce_val        = float(unreplicate(metrics['ce_loss']))
                aux_val       = float(unreplicate(metrics['aux_loss']))
                expert_load   = np.array(unreplicate(metrics['expert_load']))
    
                # Format expert load: E0:12% E1:8% ...
                expert_str = ' '.join(
                    f'E{i}:{v*100:.0f}%' for i, v in enumerate(expert_load)
                )
                print(
                    f"Epoch {epoch} | Step {global_step:06d} | "
                    f"CE {ce_val:.4f} | "
                    f"Aux {aux_val:.4f} | "
                    f"Speed {tok_per_sec:>8,.0f} tok/s | "
                    f"Elapsed {elapsed//60}m {elapsed%60:02d}s"
                )
                print(f"          Expert load: [{expert_str}]")
                last_log_time = now"""

code = code.replace(old_loop, new_loop)
with open('train.py', 'w') as f:
    f.write(code)
