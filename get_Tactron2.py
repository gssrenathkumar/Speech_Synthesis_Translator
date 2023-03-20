def has_MMI(STATE_DICT):
            return any(True for x in STATE_DICT.keys() if "mi." in x)

def get_Tactron2(MODEL_ID):
    # Download Tacotron2
    tacotron2_pretrained_model = 'MLPTTS'
    gdown.download(d+MODEL_ID, tacotron2_pretrained_model, quiet=False)
    if not exists(tacotron2_pretrained_model):
        raise Exception("Tacotron2 model failed to download!")
    # Load Tacotron2 and Config
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000 # Max Duration
    hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    state_dict = torch.load(tacotron2_pretrained_model, map_location=torch.device("cpu"))['state_dict']
    if has_MMI(state_dict):
        raise Exception("ERROR: This notebook does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.eval()
    return model, hparams

model, hparams = get_Tactron2(TACOTRON2_ID)
previous_tt2_id = TACOTRON2_ID
