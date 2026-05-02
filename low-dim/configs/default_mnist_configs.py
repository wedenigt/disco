import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 192 # 128
  training.n_iters = 500_001 # this is less than half the iterations of the cifar10 config
  training.snapshot_freq = 5_000
  training.log_freq = 100
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10_000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = False
  training.n_jitted_steps = 10 # set this as large as possible without going out of memory
  training.reduce_mean = True

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16 # typical values are 0.05 to 0.2, not sure what to pick for mnist

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.random_flip = False
  data.centered = False # this will center the data in the model, this seems to be important for both DISCO and the original NCSNv2
  data.uniform_dequantization = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 5.
  model.num_scales = 200
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.num_input_dists_per_pixel = None

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
