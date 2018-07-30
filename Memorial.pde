// Variational Auto-Encoder Implementation
// @ 2018-07-12
//
// ~ Training Process
// foreach x in X
//   epsilon := random_gaussian()
//   fp encoder
//   z := mu + epsilon * ro
//   fp encoder
//   bp decoder with L2Loss (or something else)
//   bp encoder with KLLoss
/*
int Z_DIM = 4;
FFVariable x, y, e, z, temp, kld;
FFVariable[] mu, ro, pair;
void createVAE()
{
  
  x = new FFVariable(0, 2);
  
  // Encoder
  vae_e = new FFNet();
  // < create your own encoder here >
  // temp = vae_e.builder.function(vae_e.builder.fc(x, 8), Functions.ELU);
  // < ---------------------------- >
  pair = vae_e.builder.split(temp, new int[]{Z_DIM}, new int[]{Z_DIM});
  mu = vae_e.builder.demux(pair[0], 2);
  ro = vae_e.builder.demux(vae_e.builder.function(pair[1], Functions.SQUARE), 3);
  e = new FFVariable(mu[0].priority, mu[0].x.shape);
  z = vae_e.builder.add(mu[0], vae_e.builder.mul(e, ros[0]));
  mu = ros[0];
  
  // KL-Divergence D(P||Q) = 0.5 * (mu^2 + ro^2 - log(ro^2) - 1)
  FFVariable cons = new FFVariable(mus[1].priority, mus[1].x.shape);
  add_raw(cons.x.raw(), 1, cons.x.raw());
  FFVariable kld1 = vae_e.builder.add(vae_e.builder.function(mus[1], Functions.SQUARE), ros[1]);
  FFVariable kld2 = vae_e.builder.add(vae_e.builder.function(ros[2], Functions.LOG), cons);
  kld = vae_e.builder.element_sum(vae_e.builder.sub(kld1, kld2));
  
  // Decoder
  vae_g = new FFNet();
  temp = vae_g.builder.function(vae_g.builder.fc(z, 8), Functions.TANH);
  temp = vae_g.builder.function(vae_g.builder.fc(temp, 8), Functions.TANH);
  temp = vae_g.builder.function(vae_g.builder.fc(temp, 8), Functions.TANH);
  temp = vae_g.builder.function(vae_g.builder.fc(temp, 8), Functions.TANH);
  temp = vae_g.builder.function(vae_g.builder.fc(temp, 8), Functions.TANH);
  y = vae_g.builder.function(vae_g.builder.fc(temp, 2), Functions.TANH);
  //y = vae_g.builder.fc(temp, 2);

  vae_e.init(1e-1);
  vae_g.init(1e-1);
  
  opt_e = new Adam(1e-2, 0.9, 0.999, 1e-8);
  opt_g = new Adam(1e-2, 0.9, 0.999, 1e-8);
  loss = new L2Loss();
}*/
