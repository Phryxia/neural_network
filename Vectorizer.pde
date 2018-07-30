class RGBImg2Vec
{
  /**
    Return 3-order FFCube with the shape of [w, h, 3]
  */
  public FFCube img2vec(PImage img)
  {
    img.loadPixels();
    FFCube v = new FFCube(img.width, img.height, 3);
    for(int i = 0; i < img.pixels.length; ++i)
    {
      int x = i % img.width;
      int y = i / img.height;
      color c = img.pixels[i];
      v.set(red(c)  , x, y, 0);
      v.set(green(c), x, y, 1);
      v.set(blue(c) , x, y, 2);
    }
    return v;
  }
  
  public void vec2img(FFCube v, PImage img)
  {
    if(v.axisSize(0) != img.width && v.axisSize(1) != img.height)
    {
      throw new IllegalArgumentException("[RGBImge2Vec::vec2img] Image size doesn't match to shape : " + v.axisSize(0) + " x " + v.axisSize(1) + " != " + img.width + " x " + img.height);
    }
    img.loadPixels();
    for(int y = 0; y < img.height; ++y)
    {
      for(int x = 0; x < img.width; ++x)
      {
        img.pixels[y * img.width + x] = color((float)v.get(x, y, 0), (float)v.get(x, y, 1), (float)v.get(x, y, 2));
      }
    }
    img.updatePixels();
  }
}

class GrayImg2Vec
{
  /**
    Return 3-order FFCube with the shape of [w, h, 1]
  */
  public FFCube img2vec(PImage img)
  {
    img.loadPixels();
    FFCube v = new FFCube(img.width, img.height, 1);
    for(int i = 0; i < img.pixels.length; ++i)
    {
      int x = i % img.width;
      int y = i / img.height;
      color c = img.pixels[i];
      v.set((red(c) + green(c) + blue(c))/3, x, y, 0);
    }
    return v;
  }
  
  public void vec2img(FFCube v, PImage img)
  {
    vec2img(v, img, 0);
  }
  
  public void vec2img(FFCube v, PImage img, int depth)
  {
    if(v.axisSize(0) != img.width && v.axisSize(1) != img.height)
    {
      throw new IllegalArgumentException("[GrayImge2Vec::vec2img] Image size doesn't match to shape : " + v.axisSize(0) + " x " + v.axisSize(1) + " != " + img.width + " x " + img.height);
    }
    img.loadPixels();
    for(int y = 0; y < img.height; ++y)
    {
      for(int x = 0; x < img.width; ++x)
      {
        img.pixels[y * img.width + x] = color((float)v.get(x, y, depth));
      }
    }
    img.updatePixels();
  }
}

/**
  One file has one vector.
  Useful when labeling input entries.
*/
class JSONToVec
{
  public FFCube vectorize(File file)
  {
    if(file != null)
    {
      return new FFCube(loadJSONObject(file.getAbsolutePath()));
    }
    else
    {
      throw new IllegalArgumentException("Illegal file");
    }
  }
}

/**
  Very simple linear-modeled character encoder
  shape: [str.length]
*/
class String2Vec
{
  public int encode(char c)
  {
    if('a' <= c && c <= 'z')
    {
      return (int)(c - 'a');
    }
    else
    {
      return (int)('z' - 'a' + 1);
    }
  }
  
  public char decode(int x)
  {
    if(x <= 'z' - 'a')
    {
      return (char)(x + 'a');
    }
    else
    {
      return ' ';
    }
  }
  
  public FFCube str2vec(String s)
  {
    FFCube x = new FFCube(s.length(), 'z'-'a'+2);
    for(int n = 0; n < s.length(); ++n)
    {
      int idx = encode(s.charAt(n));
      x.set(1, n, idx);
    }
    return x;
  }
  
  public String vec2str(FFCube x)
  {
    String s = "";
    for(int n = 0; n < x.axisSize(0); ++n)
    {
      int maxidx = 0;
      double maxv = 0;
      for(int i = 0; i < 'z'-'a'+2; ++i)
      {
        if(x.get(n, i) > maxv)
        {
          maxv = x.get(n, i);
          maxidx = i;
        }
      }
      s += decode(maxidx);
    }
    return s;
  }
}

// Hangul Vectorizer using NFD
