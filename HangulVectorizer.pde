/**
  One hot encoding with hangul
  First it decomposite as NFD form
  then serialize it.
*/
class HangulVectorizer
{
  public int[] BOUND; // choseong, jungseong, jongseong
  
  public HangulVectorizer()
  {
    BOUND = new int[] {19, 21, 28};
  }
  
  private char composition(int[] idx)
  {
    return (char)((idx[0] * BOUND[1] + idx[1]) * BOUND[2] + idx[2] + '\uAC00');
  }
  
  private int[] decomposition(char h)
  {
    int[] out = new int[3];
    out[2] = (h - '\uAC00') % BOUND[2];
    out[1] = (h - '\uAC00') / BOUND[2] % BOUND[1];
    out[0] = (h - '\uAC00') / BOUND[2] / BOUND[1];
    return out;
  }
  
  public FFCube str2vec(String str)
  {
    FFCube out = new FFCube(str.length(), BOUND[0] + BOUND[1] + BOUND[2]);
    for(int i = 0; i < str.length(); ++i)
    {
      int[] idx = decomposition(str.charAt(i));
      int bias = 0;
      for(int j = 0; j < 3; ++j)
      {
        out.set(1, i, bias + idx[j]);
        bias += BOUND[j];
      }
    }
    return out;
  }
  
  public String vec2str(FFCube x)
  {
    int[] idx = new int[3];
    String out = "";
    for(int i = 0; i < x.axisSize(0); ++i)
    {
      int bias = 0;
      for(int j = 0; j < 3; ++j)
      {
        double vmax = 0;
        for(int k = 0; k < BOUND[j]; ++k)
        {
          if(vmax < x.get(i, bias + k))
          {
            vmax = x.get(i, bias + k);
            idx[j] = k;
          }
        }
        bias += BOUND[j];
      }
      out += composition(idx);
    }
    return out;
  }
}
