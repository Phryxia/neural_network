// VERSION = 2018-07-12
// VERSION = 2018-07-19: add swish, delete FFBuilder
// 2018-07-30: Deprecate demux. New test example.
FFNet[] net;
FFVariable x, yhat;
FFVariable[] y;
FFVariable[] loss;
Optimizer[] opt;
ArrayList <FFCube> ts;
ArrayList <ArrayList <Double>> log = new ArrayList <ArrayList <Double>> ();
//GrayImg2Vec i2v;
RGBImg2Vec i2v;

FFNet conv_test;
FFVariable a, b, l1, l2, l3, l2loss;
Optimizer opt_test;

// Auto-Encoder config
int M = 1;
int N = 64;
int Z = 4;
int B = 16;

void setup()
{
  colorMode(RGB, 1);
  size(512, 512);
  noSmooth();
  background(0);
  /*
  x = new FFVariable(0, N, N, 1);
  yhat = new FFVariable(0, N, N, 1);
  y = new FFVariable[M];
  loss = new FFVariable[M];
  
  net = new FFNet[M];
  for(int m = 0; m < M; ++m)
  {
    createAE(m);
    if(m > 0)
      net[m].fromJSON(net[0].toJSON());
    log.add(new ArrayList <Double> ());
  }
  opt = new Optimizer[M];
  opt[0] = new AdaDelta(1e-2, 1e-8);*/
  
  i2v = new RGBImg2Vec();
  
  print("Summoning Yuigahama ");
  ts = new ArrayList <FFCube> ();
  File[] folder = (new File("D:\\dev\\dataset\\yuigahama_face\\rgb")).listFiles();
  for(File file : folder)
  {
    String path = file.getAbsolutePath();
    if(match(path, "\\.(png|jpg|jpeg)") != null)
    {
      ts.add(loadImageVector(path));
    }
    print(".");
  }
  println(" Success!");
  
  conv_test = new FFNet();
  a = new FFVariable(0, N, N, 3);
  FFVariable temp = a;
  temp = conv_test.builder.conv2d("L1", temp, 4, 4, 4, 2, 2, false); // [64, 64, 1] -> [31, 31, 4]
  l1 = temp = conv_test.builder.swish(temp);
  temp = conv_test.builder.conv2d("L2", temp, 7, 7, 8, 3, 3, false); // [31, 31, 4] -> [9, 9, 8]
  l2 = temp = conv_test.builder.swish(temp);
  //temp = conv_test.builder.fc("LF", temp, 1);
  temp = conv_test.builder.conv2d("L2", temp, 9, 9, 1, 1, 1, false); // [31, 31, 4] -> [9, 9, 8]
  temp = conv_test.builder.function(temp, Functions.TANH_01);
  
  b = new FFVariable(0, 1);
  l2loss = conv_test.builder.l2loss(temp, b);
  conv_test.builder.minimize(l2loss);
  
  //opt_test = new AdaDelta(1, 1e-8);\
  opt_test = new Nadam(1e-3, 0.9, 0.999, 1e-8);
  conv_test.init();
}



void keyReleased()
{
  if(key == 's')
  {
  }
}

void drawSubcube(FFCube ffc, int depth, float x, float y, float w, float h)
{
  PImage img = createImage(ffc.axisSize(0), ffc.axisSize(1), ARGB);
  for(int i = 0; i < ffc.axisSize(1); ++i)
  {
    for(int j = 0; j < ffc.axisSize(0); ++j)
    {
      img.set(j, i, color(0.5 + (float)ffc.get(j, i, depth)));
    }
  }
  image(img, x, y, w, h);
}

void draw()
{
  background(0);
  
  double err = 0;
  
  conv_test.beginBatch();
  a.x.set(ts.get(0));
  b.x.set(0.0, 0);
  conv_test.forward(true);
  conv_test.backward(true);
  conv_test.storeGradient();
  err += l2loss.x.get(0);
  
  a.x.set(ts.get(1));
  b.x.set(1.0, 0);
  conv_test.forward(true);
  conv_test.backward(true);
  conv_test.storeGradient();
  err += l2loss.x.get(0);
  
  conv_test.optimize(opt_test);
  println(err);
  
  // L1
  for(int d = 0; d < l1.x.axisSize(2); ++d)
  {
    float tmp = width / l1.x.axisSize(2);
    drawSubcube(l1.x, d, map(d, 0, l1.x.axisSize(2), 0, width), 0, tmp, tmp);
  }
  
  // L2
  for(int d = 0; d < l2.x.axisSize(2); ++d)
  {
    float tmp = width / l2.x.axisSize(2);
    drawSubcube(l2.x, d, map(d, 0, l2.x.axisSize(2), 0, width), height/2, tmp, tmp);
  }
  
  /*
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
  }
  
  for(int m = 0; m < M; ++m)
  {
    fill((float)m/M, 1, 1);
    int T = log.get(0).size();
    for(int t = 0; t < T; ++t)
    {
      double tmp = log.get(m).get(t);
      ellipse((float)t/T*width, map((float)tmp, 0, 3000, height, height/2), 2, 2);
    }
  }
  colorMode(RGB, 1);*/
}

FFCube loadImageVector(String file)
{
  PImage img = loadImage(file);
  img.resize(N, N);
  return i2v.img2vec(img);
}

void createAE(int id)
{
  FFNet _net = new FFNet();
  FFVariable temp = x;
  temp = _net.builder.swish(_net.builder.fc(temp, 32));
  temp = _net.builder.swish(_net.builder.fc(temp, 32));
  temp = _net.builder.swish(_net.builder.fc(temp, Z));
  temp = _net.builder.swish(_net.builder.fc(temp, 32));
  temp = _net.builder.swish(_net.builder.fc(temp, 32));
  y[id] = _net.builder.function(_net.builder.fc(temp, N, N, 1), Functions.TANH_01);
  _net.builder.minimize(loss[id] = _net.builder.l1loss(y[id], yhat));
  net[id] = _net;
  net[id].init();
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
    random_gaussian(x.x, 1e-3);
    add(x.x, td, x.x);
    clip(x.x, 0, 1);
    yhat.x.set(td);
    for(int m = 0; m < M; ++m)
    {
      net[m].forward(true);
      net[m].backward(true);
      net[m].storeGradient();
      temp[m] += loss[m].x.get(0);
    }
  }
  for(int m = 0; m < M; ++m)
  {
    log.get(m).add(temp[m] / B);
    net[m].optimize(opt[m]);
    println(m + " : " + temp[m] / B);
  }
}
