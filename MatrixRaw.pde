// 2018-07-24: Modify random_gaussian_raw to make proper variance.
// 2018-07-25: Change name and add FFCube version
// 2018-07-30: Add set(double[] md, double value) and its overloading
// 2018-07-31: Add clip
public void check_length(double[] m1, double[] m2)
{
  if(m1.length != m2.length)
    throw new IllegalArgumentException("|m1| = " + m1.length + ", |m2| = " + m2.length);
}

public void zero(double[] md)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = 0.0;
}

public void zero(FFCube md)
{
  zero(md.raw());
}

public void set(double[] md, double value)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = value;
}

public void set(FFCube md, double value)
{
  set(md.raw(), value);
}

public void upper_clip(double[] md, double max)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = Math.min(md[i], max);
}

public void upper_clip(FFCube md, double max)
{
  upper_clip(md.raw(), max);
}

public void lower_clip(double[] md, double min)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = Math.max(md[i], min);
}

public void lower_clip(FFCube md, double min)
{
  lower_clip(md.raw(), min);
}

public void clip(double[] md, double min, double max)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = Math.max(Math.min(md[i], max), min);
}

public void clip(FFCube md, double min, double max)
{
  clip(md.raw(), min, max);
}

/**
  Random Uniform
*/
public void random_uniform(double[] md, double min, double max)
{
  for(int i = 0; i < md.length; ++i)
    md[i] = Math.random() * (max - min) + min;
}

public void random_uniform(FFCube md, double min, double max)
{
  random_uniform(md.raw(), min, max);
}

/**
  Random Gaussian
*/
public void random_gaussian(double[] md, double var)
{
  var = Math.sqrt(var);
  for(int i = 0; i < md.length; ++i)
    md[i] = randomGaussian() * var;
}

public void random_gaussian(FFCube md, double var)
{
  random_gaussian(md.raw(), var);
}

/**
  Square Root
*/
public void sqrt(double[] m, double[] md)
{
  check_length(m, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = Math.sqrt(m[i]);
}

public void sqrt(FFCube m, FFCube md)
{
  sqrt(m.raw(), md.raw());
}

/**
  Copy
*/
public void copy(double[] m, double[] md)
{
  check_length(m, md);
  System.arraycopy(m, 0, md, 0, m.length);
}

/**
  Add
*/
public void add(double[] m, double s, double[] md)
{
  check_length(m, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m[i] + s;
}

public void add(FFCube m, double s, FFCube md)
{
  add(m.raw(), s, md.raw());
}

public void add(double[] m1, double[] m2, double[] md)
{
  check_length(m1, m2);
  check_length(m1, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m1[i] + m2[i];
}

public void add(FFCube m1, FFCube m2, FFCube md)
{
  add(m1.raw(), m2.raw(), md.raw());
}

/**
  Sub
*/
public void sub(double[] m, double s, double[] md)
{
  check_length(m, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m[i] - s;
}

public void sub(FFCube m, double s, FFCube md)
{
  sub(m.raw(), s, md.raw());
}

public void sub(double[] m1, double[] m2, double[] md)
{
  check_length(m1, m2);
  check_length(m1, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m1[i] - m2[i];
}

public void sub(FFCube m1, FFCube m2, FFCube md)
{
  sub(m1.raw(), m2.raw(), md.raw());
}

/**
  Multiplication
*/
public void mul(double[] m, double s, double[] md)
{
  check_length(m, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m[i] * s;
}

public void mul(FFCube m, double s, FFCube md)
{
  mul(m.raw(), s, md.raw());
}

public void mul(double[] m1, double[] m2, double[] md)
{
  check_length(m1, m2);
  check_length(m1, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m1[i] * m2[i];
}

public void mul(FFCube m1, FFCube m2, FFCube md)
{
  mul(m1.raw(), m2.raw(), md.raw());
}

/**
  Division
*/
public void div(double[] m, double s, double[] md)
{
  check_length(m, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m[i] / s;
}

public void div(FFCube m, double s, FFCube md)
{
  div(m.raw(), s, md.raw());
}

public void div(double[] m1, double[] m2, double[] md)
{
  check_length(m1, m2);
  check_length(m1, md);
  for(int i = 0; i < md.length; ++i)
    md[i] = m1[i] / m2[i];
}

public void div(FFCube m1, FFCube m2, FFCube md)
{
  div(m1.raw(), m2.raw(), md.raw());
}

/**
  Transpose
*/
public void transpose(double[] m, double[] md, int r, int c)
{
  check_length(m, md);
  for(int i = 0; i < r; ++i)
    for(int j = 0; j < c; ++j)
      md[j * r + i] = m[i * c + j];
}

// Note that this assumes m and md be R^2.
public void transpose(FFCube m, FFCube md)
{
  transpose(m.raw(), md.raw(), m.axisSize(0), m.axisSize(1));
}

/**
  Matrix Multiplication
*/
public void matmul(double[] m1, double[] m2, double[] md, int r1, int c1, int c2)
{
  if(m1.length != r1 * c1)
    throw new IllegalArgumentException("|m1| != " + r1 + " x " + c1);
  if(m2.length != c1 * c2)
    throw new IllegalArgumentException("|m2| != " + c1 + " x " + c2);
  if(md.length != r1 * c2)
    throw new IllegalArgumentException("|md| != " + r1 + " x " + c2);
  for(int i = 0; i < r1; ++i)
    for(int j = 0; j < c2; ++j)
    {
      md[i * c2 + j] = 0.0;
      for(int k = 0; k < c1; ++k)
        md[i * c2 + j] += m1[i * c1 + k] * m2[k * c2 + j];
    }
}

// Note that this assumes m1, m2 and md be R^2
public void matmul(FFCube m1, FFCube m2, FFCube md)
{
  matmul(m1.raw(), m2.raw(), md.raw(), m1.axisSize(0), m1.axisSize(1), m2.axisSize(1));
}
