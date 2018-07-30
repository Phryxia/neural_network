// VERSION = 2018-07-12
// VERSION = 2018-07-19: add swish, delete FFBuilder
// 2018-07-30: Deprecate demux. New test example.
FFNet[] net;
FFVariable x;
FFVariable[] y;
Optimizer[] opt;
ArrayList <FFCube> ts;
ArrayList <ArrayList <Double>> log = new ArrayList <ArrayList <Double>> ();
ILoss loss;
GrayImg2Vec i2v;

// Auto-Encoder config
int M = 2;
int N = 48;
int Z = 2;
int B = 32;

void setup()
{
  colorMode(RGB, 1);
  size(512, 512);
  noSmooth();
  background(0);
  
  x = new FFVariable(0, N, N, 1);
  y = new FFVariable[M];
  
  net = new FFNet[M];
  for(int m = 0; m < M; ++m)
  {
    createAE(m);
    if(m > 0)
      net[m].fromJSON(net[0].toJSON());
    log.add(new ArrayList <Double> ());
  }
  opt = new Optimizer[M];
  opt[0] = new Adam(1e-2, 0.9, 0.999, 1e-8);
  opt[1] = new Nadam(1e-2, 0.9, 0.999, 1e-8);
  
  loss = new L2Loss();
  
  i2v = new GrayImg2Vec();
  
  ts = new ArrayList <FFCube> ();
  File[] folder = (new File("D:\\dev\\dataset\\yukino_face\\rgb")).listFiles();
  for(File file : folder)
  {
    String path = file.getAbsolutePath();
    if(match(path, "\\.(png|jpg|jpeg)") != null)
    {
      ts.add(loadImageVector(path));
    }
  }
}

void createAE(int id)
{
  FFNet _net = new FFNet();
  FFVariable temp = x;
  temp = _net.builder.function(_net.builder.fc(temp, 32), Functions.LRELU);
  temp = _net.builder.function(_net.builder.fc(temp, 16), Functions.LRELU);
  temp = _net.builder.function(_net.builder.fc(temp, Z), Functions.LRELU);
  temp = _net.builder.function(_net.builder.fc(temp, 16), Functions.LRELU);
  temp = _net.builder.function(_net.builder.fc(temp, 32), Functions.LRELU);
  y[id] = _net.builder.function(_net.builder.fc(temp, N, N, 1), Functions.TANH_01);
  net[id] = _net;
  net[id].init();
}

void keyReleased()
{
  if(key == 's')
  {
  }
}

void draw()
{
  background(0);
  
  train();
  
  colorMode(HSB, 1);
  noStroke();
  x.x.set(ts.get((int)(random(0, ts.size() - 1))));
  for(int m = 0; m < M; ++m)
  {
    net[m].forward(false);
    PImage img = createImage(N, N, ARGB);
    i2v.vec2img(y[m].x, img);
    image(img, m * width / M, 0, width / M, width / M);
    
    fill((float)m/M, 1, 1);
    int T = log.get(0).size();
    for(int t = 0; t < T; ++t)
    {
      double tmp = log.get(m).get(t);
      ellipse((float)t/T*width, map((float)tmp, 0, 100, height, height/2), 2, 2);
    }
  }
  colorMode(RGB, 1);
}

FFCube loadImageVector(String file)
{
  PImage img = loadImage(file);
  img.resize(N, N);
  return i2v.img2vec(img);
}

void train()
{
  double[] temp = new double[M];
  for(int m = 0; m < M; ++m)
  {
    net[m].beginBatch();
  }
  for(int b = 0; b < B; ++b)
  {
    FFCube td = ts.get((int)(random(0, ts.size() - 1)));
    x.x.set(td);
    for(int m = 0; m < M; ++m)
    {
      net[m].forward(true);
      temp[m] += loss.loss(y[m].x, td, y[m].dx);
      net[m].backward(true);
      net[m].storeGradient();
    }
  }
  for(int m = 0; m < M; ++m)
  {
    log.get(m).add(temp[m] / B);
    net[m].optimize(opt[m]);
    println(m + " : " + temp[m] / B);
  }
}
