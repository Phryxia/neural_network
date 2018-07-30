// 2018-07-26: Add SafeLogistic
interface ILoss
{
  public double loss(FFCube y, FFCube y_desired, FFCube dy);
}

/**
  loss(y, y_desired) = 0.5 (y - y_desired)^2
*/
class L2Loss implements ILoss
{
  public double loss(FFCube y, FFCube y_desired, FFCube dy)
  {
    double err = 0.0;
    for(int i = 0; i < y.fullSize(); ++i)
    {
      double loc = y.raw()[i] - y_desired.raw()[i];
      err += 0.5 * loc * loc;
      dy.raw()[i] = loc;
    }
    return err;
  }
}

/**
  loss(y, y_desired) = 0.5 (y - y_desired)^2
*/
class L1Loss implements ILoss
{
  public double loss(FFCube y, FFCube y_desired, FFCube dy)
  {
    double err = 0.0;
    for(int i = 0; i < y.fullSize(); ++i)
    {
      double loc = y.raw()[i] - y_desired.raw()[i];
      err += Math.abs(loc);
      dy.raw()[i] = (loc > 0 ? 1 : (loc < 0 ? -1 : 0));
    }
    return err;
  }
}

class Logistic implements ILoss
{
  public double loss(FFCube y, FFCube y_desired, FFCube dy)
  {
    double err = 0.0;
    for(int i = 0; i < y.fullSize(); ++i)
    {
      double _y = y.raw()[i];
      double _yd = y_desired.raw()[i];
      err += -_yd * Math.log(_y) - (1 - _yd) * Math.log(1 - _y);
      dy.raw()[i] = -_yd / _y + (1 - _yd) / (1 - _y);
    }
    return err;
  }
}

class SafeLogistic implements ILoss
{
  private double epsilon = 1e-8;
  public double loss(FFCube y, FFCube y_desired, FFCube dy)
  {
    double err = 0.0;
    for(int i = 0; i < y.fullSize(); ++i)
    {
      double _y = Math.min(Math.max(y.raw()[i], epsilon), 1 - epsilon);
      double _yd = y_desired.raw()[i];
      err += -_yd * Math.log(_y) - (1 - _yd) * Math.log(1 - _y);
      dy.raw()[i] = -_yd / _y + (1 - _yd) / (1 - _y);
    }
    return err;
  }
}
