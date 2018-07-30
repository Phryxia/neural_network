/**
  FFCube represent high order data
  which is handled by FFNet
  This can be serialized into Matrix
*/
static class FFCube implements JSONAble
{
  private int[]    shape;
  private double[] value;
  
  /**
    Create [dt] size's cube with
    given each sizes [d]
  */
  public FFCube(int ... d)
  {
    shape = new int[d.length];
    System.arraycopy(d, 0, shape, 0, d.length);
    value = new double[computeTotalSize()];
  }
  
  /**
    Scan the shape array and return the
    total size of this ffcube.
  */
  private int computeTotalSize()
  {
    int factor = 1;
    for(int i = 0; i < order(); ++i)
    {
      factor *= shape[i];
    }
    return factor;
  }
  
  /**
    Construct FFCube from given JSONObject.
  */
  public FFCube(JSONObject json)
  {
    fromJSON(json);
  }
  
  /**
    Return the JSONObject representation.
  */
  public JSONObject toJSON()
  {
    JSONObject json = new JSONObject();
    json.setJSONArray("shape", toJSONArray(shape));
    json.setJSONArray("value", toJSONArray(value));
    return json;
  }
  
  /**
    Parse JSONObject
  */
  public void fromJSON(JSONObject json)
  {
    shape = json.getJSONArray("shape").getIntArray();
    value = json.getJSONArray("value").getDoubleArray();
  }
  
  /**
    Return the shape array of this ffcube.
  */
  public int[] shape()
  {
    return shape;
  }
  
  /**
    Return the array representation of data.
    Every change to the return value affect
    to original.
  */
  public double[] raw()
  {
    return value;
  }
  
  /**
    Return the number of axis in this cube.
    For example, 2 x 2 x 2 ffcube's order is 3.
  */
  public int order()
  {
    return shape.length;
  }
  
  /**
    Return the 
  */
  public int axisSize(int d)
  {
    return shape[d];
  }
  
  /**
    Return the number of variables in this cube.
    This is equivalent to getRawValues().length.
  */
  public int fullSize()
  {
    return value.length;
  }
  
  /**
    Encode the tuple representation to the 1D index.
  */
  public int encode(int ... crd)
  {
    if(crd.length != order())
    {
      throw new IllegalArgumentException("[FFCube::encode] Number of parameters(" + crd.length + ") doesn't match to cube's order(" + order() + ")");
    }
    int idx = 0;
    for(int i = 0; i < order(); ++i)
    {
      idx = idx * axisSize(i) + crd[i];
    }
    return idx;
  }
  
  /**
    Return given position's value
  */
  public double get(int ... crd)
  {
    return value[encode(crd)];
  }
  
  /**
    Set given position's value to val and return itself.
  */
  public FFCube set(double val, int ... crd)
  {
    value[encode(crd)] = val;
    return this;
  }
  
  /**
    Set every values of cube as given and return itself.
  */
  public FFCube set(FFCube c)
  {
    System.arraycopy(c.value, 0, value, 0, value.length);
    return this;
  }
}
