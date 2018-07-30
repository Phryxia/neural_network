// 2018-07-24 version: apply Initializer interface.
// 2018-07-26: swish now initialize beta with random gaussian.
// 2018-07-27: binary_cross_entropy!
/**
  FFVariable is a pair of FFCube which
  holds vector and its gradient.
*/
class FFVariable
{
  public final int priority;
  public FFCube x, dx;
  public Initializer initializer;
  
  public FFVariable(int _priority, Initializer _initializer, int ... shape)
  {
    priority = _priority;
    x = new FFCube(shape);
    dx = new FFCube(shape);
    initializer = _initializer;
  }
  
  public FFVariable(int _priority, int ... shape)
  {
    priority = _priority;
    x = new FFCube(shape);
    dx = new FFCube(shape);
    initializer = new ConstantInit(0.0);
  }
}

class FFParam extends FFVariable
{
  public FFParam(int priority, int ... shape)
  {
    super(priority, shape);
  }
}

class FFNet
{
  HashMap <String, FFVariable> search;
  ArrayList <FFVariable> parameters;
  ArrayList <ArrayList <FFModule>> layer;
  double[] serial_p;
  double[] serial_g;
  Optimizer optimizer;
  int cnt = 0;
  
  public final Builder builder;
  
  public FFNet()
  {
    search = new HashMap <String, FFVariable> ();
    parameters = new ArrayList <FFVariable> ();
    layer = new ArrayList <ArrayList <FFModule>> ();
    cnt = 0;
    
    builder = new Builder();
  }
  
  protected void addModule(FFModule module, int priority)
  {
    while(layer.size() - 1 < priority)
      layer.add(new ArrayList <FFModule> ());
    layer.get(priority).add(module);
  }
  
  public void addParameter(FFVariable ffv, String name)
  {
    search.put(name, ffv);
    parameters.add(ffv);
  }
  
  public void init()
  {
    int len = 0;
    for(FFVariable ffv : parameters)
    {
      len += ffv.x.fullSize();
      ffv.initializer.initialize(ffv.x);
    }
    serial_p = new double[len];
    serial_g = new double[len];
  }
  
  public void beginBatch()
  {
    zero(serial_g);
    cnt = 0;
  }
  
  public void forward(boolean isTraining)
  {
    for(ArrayList <FFModule> ly : layer)
      for(FFModule module : ly)
        module.forward(isTraining);
  }
  
  public void backward(boolean isTraining)
  {
    for(ArrayList <FFModule> ly : layer)
      for(FFModule module : ly)
        for(int n = 0; n < module.ipin_capacity; ++n)
          zero(module.getIPin(n).dx);
    for(int i = layer.size() - 1; i >= 0; --i)
      for(FFModule module : layer.get(i))
        module.backward(isTraining);
  }
  
  public void storeGradient()
  {
    int ptr = 0;
    for(FFVariable ffv : parameters)
      for(int i = 0 ; i < ffv.dx.fullSize(); ++i)
        serial_g[ptr++] += ffv.dx.raw()[i];
    ++cnt;
  }
  
  public void optimize(Optimizer optimizer)
  {
    if(this.optimizer == null)
    {
      this.optimizer = optimizer;
      optimizer.init(serial_p.length);
    }
    div(serial_g, cnt, serial_g);
    param2serial();
    optimizer.optimize(serial_p, serial_g);
    serial2param();
  }
  
  private void param2serial()
  {
    int ptr = 0;
    for(FFVariable ffv : parameters)
    {
      System.arraycopy(ffv.x.raw(), 0, serial_p, ptr, ffv.x.fullSize());
      ptr += ffv.x.fullSize();
    }
  }
  
  private void serial2param()
  {
    int ptr = 0;
    for(FFVariable ffv : parameters)
    {
      System.arraycopy(serial_p, ptr, ffv.x.raw(), 0, ffv.x.fullSize());
      ptr += ffv.x.fullSize();
    }
  }
  
  public JSONArray toJSON()
  {
    param2serial();
    return toJSONArray(serial_p);
  }
  
  public void fromJSON(JSONArray json)
  {
    serial_p = json.getDoubleArray();
    serial2param();
  }
  
  /**
    Code is generated as following:
    nodename_paramname
  */
  public FFVariable getParameter(String code)
  {
    return search.get(code);
  }
  
  private class Builder
  {
    public FFVariable add(FFVariable ... ffvs)
    {
      // Scan max priority
      int max_priority = ffvs[0].priority;
      for(int i = 1; i < ffvs.length; ++i)
        max_priority = Math.max(max_priority, ffvs[i].priority);
        
      // Connect topology
      FFModule module = new FFAdd(ffvs.length, max_priority);
      for(int i = 0; i < ffvs.length; ++i)
        module.setIPin(i, ffvs[i]);
      FFVariable out = new FFVariable(max_priority + 1, ffvs[0].x.shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, max_priority);
      
      return out;
    }
    
    public FFVariable add(FFVariable ffv, double scalar)
    {
      FFVariable c = new FFVariable(0, ffv.x.shape);
      for(int i = 0; i < c.x.fullSize(); ++i)
        c.x.raw()[i] = scalar;
      return add(ffv, c);
    }
    
    public FFVariable sub(FFVariable ffv1, FFVariable ffv2)
    {
      // Scan max priority
      int priority = Math.max(ffv1.priority, ffv2.priority);
        
      // Connect topology
      FFModule module = new FFSub(priority);
      module.setIPin(0, ffv1);
      module.setIPin(1, ffv2);
      FFVariable out = new FFVariable(priority + 1, ffv1.x.shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    public FFVariable sub(FFVariable ffv, double scalar)
    {
      FFVariable c = new FFVariable(0, ffv.x.shape);
      for(int i = 0; i < c.x.fullSize(); ++i)
        c.x.raw()[i] = scalar;
      return sub(ffv, c);
    }
    
    public FFVariable sub(double scalar, FFVariable ffv)
    {
      FFVariable c = new FFVariable(0, ffv.x.shape);
      for(int i = 0; i < c.x.fullSize(); ++i)
        c.x.raw()[i] = scalar;
      return sub(c, ffv);
    }
    
    public FFVariable mul(FFVariable ffv1, FFVariable ffv2)
    {
      // Scan max priority
      int priority = Math.max(ffv1.priority, ffv2.priority);
        
      // Connect topology
      FFModule module = new FFMul(priority);
      module.setIPin(0, ffv1);
      module.setIPin(1, ffv2);
      FFVariable out = new FFVariable(priority + 1, ffv1.x.shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    public FFVariable mul(FFVariable ffv, double scalar)
    {
      FFVariable c = new FFVariable(0, ffv.x.shape);
      for(int i = 0; i < c.x.fullSize(); ++i)
        c.x.raw()[i] = scalar;
      return mul(ffv, c);
    }
    
    public FFVariable element_sum(FFVariable ffv)
    {
      // Scan max priority
      int priority = ffv.priority;
        
      // Connect topology
      FFModule module = new FFElemSum(priority);
      module.setIPin(0, ffv);
      FFVariable out = new FFVariable(priority + 1, 1);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    public FFVariable function(FFVariable ffv, IFunction f)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFModule module = new FFFunction(priority, f);
      module.setIPin(0, ffv);
      
      FFVariable out = new FFVariable(priority + 1, ffv.x.shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    public FFVariable fc(String name, FFVariable ffv, Initializer W_init, Initializer b_init, int ... oshape)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFModule module = new FFFullConn(priority);
      module.setIPin(0, ffv);
      
      FFVariable out = new FFVariable(priority + 1, oshape);
      module.setOPin(0, out);
      
      FFVariable W = new FFVariable(priority, W_init, out.x.fullSize(), ffv.x.fullSize());
      module.setIPin(1, W);
      
      FFVariable b = new FFVariable(priority, b_init, oshape);
      module.setIPin(2, b);
      
      // Assign module and parameters
      addModule(module, priority);
      addParameter(W, name + "_W");
      addParameter(b, name + "_b");
      
      return out;
    }
    
    public FFVariable fc(String name, FFVariable ffv, int ... oshape)
    {
      return fc(name, ffv, new He(), new ConstantInit(0), oshape);
    }
    
    public FFVariable fc(FFVariable ffv, int ... oshape)
    {
      return fc("", ffv, new He(), new ConstantInit(0), oshape);
    }
    
    public FFVariable drop(FFVariable ffv, double rate)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFModule module = new FFDrop(priority, rate);
      module.setIPin(0, ffv);
      
      FFVariable out = new FFVariable(priority + 1, ffv.x.shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    public FFVariable[] split(FFVariable ffv, int[] oshape1, int[] oshape2)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFModule module = new FFSplit(priority);
      module.setIPin(0, ffv);
      
      FFVariable out1 = new FFVariable(priority + 1, oshape1);
      module.setOPin(0, out1);
      
      FFVariable out2 = new FFVariable(priority + 1, oshape2);
      module.setOPin(1, out2);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return new FFVariable[]{out1, out2};
    }
    
    public FFVariable join(FFVariable ... ffvs)
    {
      // Define priority
      int shape = ffvs[0].x.fullSize();
      int priority = ffvs[0].priority;
      for(int i = 1; i < ffvs.length; ++i)
      {
        priority = Math.max(priority, ffvs[i].priority);
        shape += ffvs[i].x.fullSize();
      }
      
      // Connect topology
      FFModule module = new FFJoin(ffvs.length, priority);
      for(int i = 0; i < ffvs.length; ++i)
        module.setIPin(i, ffvs[i]);
      
      FFVariable out = new FFVariable(priority + 1, shape);
      module.setOPin(0, out);
      
      // Assign module and parameters
      addModule(module, priority);
      
      return out;
    }
    
    /*
    public FFVariable[] demux(FFVariable ffv, int count)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFModule module = new FFDemux(count, priority);
      module.setIPin(0, ffv);
      
      FFVariable[] outs = new FFVariable[count];
      for(int i = 0; i < count; ++i)
      {
        outs[i] = new FFVariable(priority + 1, ffv.x.shape);
        module.setOPin(i, outs[i]);
      }
      
      // Assign module and parameters
      addModule(module, priority);
      
      return outs;
    }*/
    
    public FFVariable swish(String name, FFVariable ffv)
    {
      // Define priority
      int priority = ffv.priority;
      
      // Connect topology
      FFVariable beta = new FFVariable(ffv.priority, new GaussianInit(2), ffv.x.shape);
      //FFVariable[] ffvs = this.demux(ffv, 2);
      FFVariable out = this.mul(ffv, this.function(this.mul(ffv, beta), Functions.SIGMOID));
      
      // Assign module and parameters
      addParameter(beta, name + "_beta");
      
      return out;
    }
    
    public FFVariable swish(FFVariable ffv)
    {
      return swish("", ffv);
    }
    
    // Alias for frequently used mathematic functions
    public FFVariable exp(FFVariable ffv)
    {
      return function(ffv, Functions.EXP);
    }
    
    public FFVariable log(FFVariable ffv)
    {
      return function(ffv, Functions.LOG);
    }
    
    public FFVariable square(FFVariable ffv)
    {
      return function(ffv, Functions.SQUARE);
    }
    
    public FFVariable sqrt(FFVariable ffv)
    {
      return function(ffv, Functions.SQRT);
    }
    
    /**
      xhat: estimated
      x   : desired
    */
    public FFVariable binary_cross_entropy(FFVariable xhat, FFVariable x)
    {
      //FFVariable[] xhats = demux(xhat, 2);
      //FFVariable[] xs = demux(x, 2);
      return element_sum(mul(add(
        mul(x, log(xhat)),
        mul(sub(1, x), log(sub(1, xhat)))
      ), -1));
    }
  }
}
