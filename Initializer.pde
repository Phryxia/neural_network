// 2018-07-24 version: Created.
// 2018-07-26: Add Gaussian and Uniform.
/**
  Initializer interface provides simple
  parameter initializing method.
  
  Note that they are not always working:
  they do when they're 2nd order hypercube.
  (For example, the parameter used in full
  connection: which holds for input x output)

  Reference
  http://www.khshim.com/archives/641
*/
interface Initializer
{
  public abstract void initialize(FFCube p);
}

/**
  Super simple initializer algorithm.
  
  Please use this only for Bias, if you
  don't have any good reason.
  
  This support every shape of FFCubes.
*/
class ConstantInit implements Initializer
{
  private double value;
  public ConstantInit(double value)
  {
    this.value = value;
  }
  
  public void initialize(FFCube p)
  {
    for(int i = 0; i < p.fullSize(); ++i)
      p.raw()[i] = value;
  }
}

class GaussianInit implements Initializer
{
  private double var;
  public GaussianInit(double var)
  {
    this.var = var;
  }
  
  public void initialize(FFCube p)
  {
    random_gaussian(p, var);
  }
}

class UniformInit implements Initializer
{
  private double min, max;
  public UniformInit(double min, double max)
  {
    this.min = min;
    this.max = max;
  }
  
  public void initialize(FFCube p)
  {
    random_uniform(p, min, max);
  }
}

/**
  LeCun initialization is good old method
  proposed by LeCun at 1998.
  
  This works badly when used at ReLU-like
  activation functions.
  
  This support FFCubes having order() >= 2 only
*/
class LeCunUniform implements Initializer
{
  public void initialize(FFCube p)
  {
    if(p.order() < 2)
      throw new IllegalArgumentException("LeCunUniform doesn't support 1st order FFCube.");
    double bound = Math.sqrt(3.0 / p.axisSize(1));
    random_uniform(p, -bound, bound);
  }
}

class LeCunGaussian implements Initializer
{
  public void initialize(FFCube p)
  {
    if(p.order() < 2)
      throw new IllegalArgumentException("LeCunGaussian doesn't support 1st order FFCube.");
    random_gaussian(p, 1.0 / p.axisSize(1));
  }
}

/**
  Glorot initialization is proposed by
  Glorot and Bengio at 2010.
  
  This works fine with ReLU like friends,
  but not the best.
  
  This support FFCubes having order() >= 2 only
*/
class GlorotUniform implements Initializer
{
  public void initialize(FFCube p)
  {
    if(p.order() < 2)
      throw new IllegalArgumentException("GlorotUniform doesn't support 1st order FFCube.");
    double bound = Math.sqrt(6.0 / (p.axisSize(0) + p.axisSize(1)));
    random_uniform(p, -bound, bound);
  }
}

class GlorotGaussian implements Initializer
{
  public void initialize(FFCube p)
  {
    if(p.order() < 2)
      throw new IllegalArgumentException("GlorotGaussian doesn't support 1st order FFCube.");
    random_gaussian(p, 2.0 / (p.axisSize(0) + p.axisSize(1)));
  }
}

/**
  He initialization is proposed by
  he at 2015.
  
  This beat GoogleNet using PReLU
  
  This support FFCubes having order() >= 2 only
*/
class He implements Initializer
{
  public void initialize(FFCube p)
  {
    if(p.order() < 2)
      throw new IllegalArgumentException("He doesn't support 1st order FFCube.");
    random_gaussian(p, 2.0 / p.axisSize(1));
  }
}
