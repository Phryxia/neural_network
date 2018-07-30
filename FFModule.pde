// 2018-07-29: Change overwriting policy to adding policy.
// 2018-07-30: Add StaticFeeder
/**
  FFModule is a micro arithmetic unit
  which can perform forward/backward
  operation on several pins.
  
  FFModule must add its gradient result
  to the faucet!
  
  FFModule has operation priority. Computation order of
  FFModules having same priority is undefined.
  
  FFModule also has input/output pins and
  they own their unique index.
  
  We don't recommend to instanciate FFModule's child
  directly since it requires several boring connection.
  You can acquire them via FFNet.builder
*/
abstract class FFModule
{
  // priority is a computation order in FFNet.
  public final int priority;
  
  // capacity must be larger than 0
  public final int ipin_capacity;
  public final int opin_capacity;
  private FFVariable[] ipin;
  private FFVariable[] opin;
  
  // Constructor with given ipin/opin capacity and the priorty
  public FFModule(int _ipin_capacity, int _opin_capacity, int _priority)
  {
    priority = _priority;
    ipin_capacity = _ipin_capacity;
    opin_capacity = _opin_capacity;
    ipin = new FFVariable[ipin_capacity];
    opin = new FFVariable[opin_capacity];
  }
  
  // Connect ipin at given position.
  public void setIPin(int index, FFVariable ffv)
  {
    ipin[index] = ffv;
  }
  
  // Return given position's ipin
  public FFVariable getIPin(int index)
  {
    return ipin[index];
  }
  
  // Connect opin at given position.
  public void setOPin(int index, FFVariable ffv)
  {
    opin[index] = ffv;
  }
  
  // Return given position's opin
  public FFVariable getOPin(int index)
  {
    return opin[index];
  }
  
  // The behavior when FFNet's forward is called.
  public abstract void forward(boolean isTraining);
  
  // The behavior when FFNet's backward is called.
  public abstract void backward(boolean isTraining);
  
  // The behavior when FFNet's beginBatch is called.
  public abstract void batch();
}

/**
  <forward>
    y := x0 + x1 + ... + xn
    
  <backward>
    dx0, ..., dxn += dy
    
  <ipin>
    0: x0
    ...
    n: xn
  
  <opin>
    0: y
*/
class FFAdd extends FFModule
{
  public FFAdd(int ipin_capacity, int priority)
  {
    super(ipin_capacity, 1, priority);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    FFCube y = getOPin(0).x;
    y.set(getIPin(0).x);
    for(int i = 1; i < ipin_capacity; ++i)
      add(getIPin(i).x, y, y);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    FFCube dy = getOPin(0).dx;
    for(int i = 0; i < ipin_capacity; ++i)
      add(dy, getIPin(i).dx, getIPin(i).dx);
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  <forward>
    y := a - b
    
  <backward>
    da += dy
    db += -dy
    
  <ipin>
    0: a
    1: b
  
  <opin>
    0: y
*/
class FFSub extends FFModule
{
  public FFSub(int priority)
  {
    super(2, 1, priority);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    sub(getIPin(0).x, getIPin(1).x, getOPin(0).x);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    add(getIPin(0).dx, getOPin(0).dx, getIPin(0).dx);
    sub(getIPin(1).dx, getOPin(0).dx, getIPin(1).dx);
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  <forward>
    y := a .* b
    
  <backward>
    da += dy .* b
    db += dy .* a
    
  <ipin>
    0: a
    1: b
  
  <opin>
    0: y
*/
class FFMul extends FFModule
{
  private FFCube temp;
  
  public FFMul(int priority)
  {
    super(2, 1, priority);
  }
  
  @Override
  public void setOPin(int index, FFVariable ffv)
  {
    super.setOPin(index, ffv);
    temp = new FFCube(ffv.x.shape);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    mul(getIPin(0).x, getIPin(1).x, getOPin(0).x);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    mul(getIPin(1).x, getOPin(0).dx, temp);
    add(getIPin(0).dx, temp, getIPin(0).dx);
    mul(getIPin(0).x, getOPin(0).dx, temp);
    add(getIPin(1).dx, temp, getIPin(1).dx);
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  <forward>
    y := x[0] + ... + x[n]
    
  <backward>
    dx[0], ..., dx[n] += dy
    
  <ipin>
    0: x
  
  <opin>
    0: y
*/
class FFElemSum extends FFModule
{
  public FFElemSum(int priority)
  {
    super(1, 1, priority);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] y = getOPin(0).x.raw();
    y[0] = 0.0;
    for(int i = 0; i < x.length; ++i)
      y[0] += x[i];
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    double[] dx = getIPin(0).dx.raw();
    double[] dy = getOPin(0).dx.raw();
    for(int i = 0; i < dx.length; ++i)
      dx[i] += dy[0];
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  order(W)  = 2
  dim(W)    = (sizeof(s), sizeof(x))
  sizeof(b) = sizeof(s)
  
  <forward>
    s := Wx + b
  
  <backward>
    dx := W^t * ds
    dW := ds  * x^t
    db := ds

  <ipin>
    0: x
    1: W
    2: b
  
  <opin>
    0: s
*/
class FFFullConn extends FFModule
{
  private double[] xt, Wt;
  
  public FFFullConn(int priority)
  {
    super(3, 1, priority);
  }
  
  @Override
  public void setIPin(int index, FFVariable ffv)
  {
    super.setIPin(index, ffv);
    switch(index)
    {
      case 0:
        xt = new double[ffv.x.fullSize()];
        break;
      case 1:
        Wt = new double[ffv.x.fullSize()];
        break;
    }
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] W = getIPin(1).x.raw();
    double[] b = getIPin(2).x.raw();
    double[] s = getOPin(0).x.raw();
    matmul(W, x, s, s.length, x.length, 1);
    add(b, s, s);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] W = getIPin(1).x.raw();
    double[] dx = getIPin(0).dx.raw();
    double[] dW = getIPin(1).dx.raw();
    double[] db = getIPin(2).dx.raw();
    double[] ds = getOPin(0).dx.raw();
    int Wrow = getIPin(1).x.axisSize(0);
    int Wcol = getIPin(1).x.axisSize(1);
    
    // dx += W^t * dy, Use xt as temporary
    transpose(W, Wt, Wrow, Wcol);
    matmul(Wt, ds, xt, Wcol, Wrow, 1);
    add(dx, xt, dx);
    
    // dW += ds * x^t, Use Wt as temporary
    transpose(x, xt, Wcol, 1);
    matmul(ds, xt, Wt, Wrow, 1, Wcol);
    add(dW, Wt, dW);
    
    // db += ds
    add(db, ds, db);
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  <forward>
    s := f(x)
  
  <backward>
    dx := f'(x) .* ds
  
  <ipin>
    0: x
  
  <opin>
    0: s
*/
class FFFunction extends FFModule
{
  private IFunction f;
  
  public FFFunction(int priority, IFunction function)
  {
    super(1, 1, priority);
    f = function;
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] s = getOPin(0).x.raw();
    for(int i = 0; i < x.length; ++i)
      s[i] = f.f(x[i]);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    double[] x  = getIPin(0).x.raw();
    double[] dx = getIPin(0).dx.raw();
    double[] s  = getOPin(0).x.raw();
    double[] ds = getOPin(0).dx.raw();
    for(int i = 0; i < x.length; ++i)
      dx[i] += f.dfdx(x[i], s[i]) * ds[i];
  }
  
  @Override
  public void batch()
  {
  }
}


/**
  <forward>
    s := f(x)
  
  <backward>
    dx := f'(x) .* ds
  
  <ipin>
    0: x
  
  <opin>
    0: s
    
  !Warning!
  When backward() is executed, temporary value
  will be reset. Therefore if you call that function
  more than once directly, something would get wrong.
*/
class FFDrop extends FFModule
{
  private double rate;
  private double[] temp;
  
  public FFDrop(int priority, double rate)
  {
    super(1, 1, priority);
    this.rate = rate;
  }
  
  @Override
  public void setIPin(int index, FFVariable ffv)
  {
    super.setIPin(index, ffv);
    temp = new double[ffv.x.fullSize()];
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] s = getOPin(0).x.raw();
    if(isTraining)
    {
      random_uniform(temp, 0, 1);
      for(int i = 0; i < temp.length; ++i)
        temp[i] = temp[i] > rate ? 1 : 0;
      mul(x, temp, s);
    }
    else
    {
      copy(x, s);
    }
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    double[] dx = getIPin(0).dx.raw();
    double[] ds = getOPin(0).dx.raw();
    if(isTraining)
    {
      mul(ds, temp, temp);
      add(dx, temp, dx);
    }
    else
    {
      add(dx, ds, dx);
    }
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  Split one vector to small vectors having
  [s's fullSize()] [input's fullSize() - s's fullSize()]
*/
class FFSplit extends FFModule
{
  public FFSplit(int priority)
  {
    super(1, 2, priority);
  }
  
  public void setOPin(int index, FFVariable ffv)
  {
    super.setOPin(index, ffv);
    if(index == 1 && getIPin(0).x.fullSize() != getOPin(0).x.fullSize() + getOPin(1).x.fullSize())
      throw new IllegalArgumentException("Split size mismatched: "
        + getOPin(0).x.fullSize() + " + " + getOPin(1).x.fullSize() + " != " + getIPin(0).x.fullSize());
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    double[] x = getIPin(0).x.raw();
    double[] s = getOPin(0).x.raw();
    double[] t = getOPin(1).x.raw();
    System.arraycopy(x, 0, s, 0, s.length);
    System.arraycopy(x, s.length, t, 0, t.length);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    double[] dx = getIPin(0).dx.raw();
    double[] ds = getOPin(0).dx.raw();
    double[] dt = getOPin(1).dx.raw();
    for(int i = 0; i < ds.length; ++i)
      dx[i] += ds[i];
    for(int i = 0; i < dt.length; ++i)
      dx[ds.length + i] += dt[i];
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  Join several cubes into a cube.
  This will destruct original shape.
*/
class FFJoin extends FFModule
{
  public FFJoin(int ipin_capacity, int priority)
  {
    super(ipin_capacity, 1, priority);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    int bias = 0;
    double[] s = getOPin(0).x.raw();
    for(int i = 0; i < ipin_capacity; ++i)
    {
      double[] x = getIPin(i).x.raw();
      System.arraycopy(x, 0, s, bias, x.length);
      bias += x.length;
    }
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    int bias = 0;
    double[] ds = getOPin(0).dx.raw();
    for(int i = 0; i < ipin_capacity; ++i)
    {
      double[] dx = getIPin(i).dx.raw();
      for(int j = 0; j < dx.length; ++j)
        ds[bias + j] += dx[j];
      bias += dx.length;
    }
  }
  
  @Override
  public void batch()
  {
  }
}

public class StaticFeeder extends FFModule
{
  private double value;
  
  public StaticFeeder(int priority, double value)
  {
    super(1, 0, priority);
    this.value = value;
  }
  
  @Override
  public void forward(boolean isTraining)
  {
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    set(getIPin(0).dx, value);
  }
  
  @Override
  public void batch()
  {
  }
}

/**
  Demux: copy one vectors to others
*/
/*
class FFDemux extends FFModule
{
  public FFDemux(int opin_capacity, int priority)
  {
    super(1, opin_capacity, priority);
  }
  
  @Override
  public void forward(boolean isTraining)
  {
    for(int i = 0; i < opin_capacity; ++i)
      getOPin(i).x.set(getIPin(0).x);
  }
  
  @Override
  public void backward(boolean isTraining)
  {
    getIPin(0).dx.set(getOPin(0).dx);
    for(int i = 1; i < opin_capacity; ++i)
      add(getOPin(i).dx, getIPin(0).dx, getIPin(0).dx);
  }
  
  @Override
  public void batch()
  {
  }
}*/
