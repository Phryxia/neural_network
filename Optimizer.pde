// 2018-07-30: add Momentum and NAG

interface Optimizer
{
  public abstract void init(int size);
  public abstract void optimize(double[] x, double[] dx);
}

/*
public class SGD implements Optimizer
{
}*/

public class Momentum implements Optimizer
{
  private double[] v, t;
  private double alpha, beta;
  
  public Momentum
  (
    double alpha,
    double beta
  )
  {
    this.alpha = alpha;
    this.beta = beta;
  }
  
  public void init(int size)
  {
    v = new double[size];
    t = new double[size];
  }
  
  public void optimize(double[] x, double[] dx)
  {
    // v := beta * v - alpha * dx
    // x := x - v
    mul(dx, alpha, t);
    mul(v, beta, v);
    sub(v, t, v);
    add(x, v, x);
  }
}

public class NAG implements Optimizer
{
  private double[] v, vp;
  private double[] t;
  private double alpha, beta;
  
  public NAG
  (
    double alpha,
    double beta
  )
  {
    this.alpha = alpha;
    this.beta = beta;
  }
  
  public void init(int size)
  {
    vp = new double[size];
    v = new double[size];
    t = new double[size];
  }
  
  // referene: https://tensorflow.blog/2017/03/22/momentum-nesterov-momentum/
  public void optimize(double[] x, double[] dx)
  {
    copy(v, vp);
    mul(v, beta, v);
    mul(dx, alpha, t);
    sub(v, t, v);
    mul(vp, beta, vp);
    mul(v, 1 + beta, t);
    sub(x, vp, x);
    add(x, t, x);
  }
}

public class Adam implements Optimizer
{
  private double[] m, v, t;
  private double alpha, beta1, beta2, beta1c, beta2c, epsilon;
  
  public Adam
  (
    double alpha,
    double beta1,
    double beta2,
    double epsilon
  )
  {
    this.alpha = alpha;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }
  
  public void init(int size)
  {
    m = new double[size];
    v = new double[size];
    t = new double[size];
    beta1c = beta1;
    beta2c = beta2;
  }
  
  public void optimize(double[] x, double[] dx)
  {
    // m := b1 * m + (1 - b1) * dx
    mul(m, beta1, m);
    mul(dx, 1 - beta1, t);
    add(m, t, m);
    
    // v := b2 * v + (1 - b2) * dx^2
    mul(v, beta2, v);
    mul(dx, dx, t);
    mul(t, 1 - beta2, t);
    add(v, t, v);
    
    // x := x - alpha * m / sqrt(v + epsilon)
    div(v, 1 - beta2c, t);
    add(t, epsilon, t);
    sqrt(t, t);
    div(m, t, t);
    mul(t, alpha, t);
    div(t, 1 - beta1c, t);
    sub(x, t, x);
    
    beta1c *= beta1;
    beta2c *= beta2;
  }
}

public class AdaMax implements Optimizer
{
  private double[] m, v, t;
  private double alpha, beta1, beta2, beta1c, epsilon;
  
  public AdaMax
  (
    double alpha,
    double beta1,
    double beta2,
    double epsilon
  )
  {
    this.alpha = alpha;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }
  
  public void init(int size)
  {
    m = new double[size];
    v = new double[size];
    t = new double[size];
    beta1c = beta1;
  }
  
  public void optimize(double[] x, double[] dx)
  {
    // m := b1 * m + (1 - b1) * dx
    mul(m, beta1, m);
    mul(dx, 1 - beta1, t);
    add(m, t, m);
    
    // v := max(b2 * v, |dx|)
    for(int i = 0; i < dx.length; ++i)
      v[i] = Math.max(beta2 * v[i], Math.abs(dx[i]));
    
    // x := x - alpha * m / sqrt(v + epsilon)
    add(v, epsilon, t);
    sqrt(t, t);
    div(m, t, t);
    mul(t, alpha, t);
    div(t, 1 - beta1c, t);
    sub(x, t, x);
    
    beta1c *= beta1;
  }
}

// refrence: http://cs229.stanford.edu/proj2015/054_report.pdf
public class Nadam implements Optimizer
{
  private double[] m, g, mh, gh, n, nh, t;
  private double alpha, beta1, beta2, beta1c, beta2c, epsilon;
  
  public Nadam
  (
    double alpha,
    double beta1,
    double beta2,
    double epsilon
  )
  {
    this.alpha = alpha;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }
  
  public void init(int size)
  {
    m = new double[size];
    mh = new double[size];
    g = new double[size];
    gh = new double[size];
    n = new double[size];
    nh = new double[size];
    t = new double[size];
    beta1c = beta1;
    beta2c = beta2;
  }
  
  public void optimize(double[] x, double[] dx)
  {
    // g := dx
    copy(dx, g);
    
    // gh := g / (1 - beta1^t)
    div(g, 1 - beta1c, gh);
    
    // m := beta1 * m + (1 - beta1) * g
    mul(m, beta1, m);
    mul(g, 1 - beta1, t);
    add(m, t, m);
    
    // mh := m / (1 - beta1^t)
    div(m, 1 - beta1c, mh);
    
    // n := beta2 * n + (1 - beta2) * g^2
    mul(n, beta2, n);
    mul(g, g, t);
    mul(t, 1 - beta2, t);
    add(n, t, n);
    
    // nh := n / (1 - beta2^t)
    div(n, 1 - beta2c, nh);
    
    // m_ := (1 - beta1) * gh + beta * mh
    mul(gh, 1 - beta1, gh);
    mul(mh, beta1, mh);
    add(gh, mh, t);
    
    // x := x - alpha * m_ / (sqrt(nh) + epsilon)
    mul(t, alpha, t);
    sqrt(nh, nh);
    add(nh, epsilon, nh);
    div(t, nh, t);
    sub(x, t, x);
    
    beta1c *= beta1;
    beta2c *= beta2;
  }
}
