// 2018-07-26: Add random_index
interface JSONAble
{
  public JSONObject toJSON();
  public void fromJSON(JSONObject json);
}

static JSONArray toJSONArray(int[] darr)
{
  JSONArray out = new JSONArray();
  for(int i = 0; i < darr.length; ++i)
  {
    out.setInt(i, darr[i]);
  }
  return out;
}

static JSONArray toJSONArray(double[] darr)
{
  JSONArray out = new JSONArray();
  for(int i = 0; i < darr.length; ++i)
  {
    out.setDouble(i, darr[i]);
  }
  return out;
}

static JSONArray toJSONArray(ArrayList <? extends JSONAble> list)
{
  JSONArray array = new JSONArray();
  for(int i = 0; i < list.size(); ++i)
  {
    array.setJSONObject(i, list.get(i).toJSON());
  }
  return array;
}

class Pair <L, R>
{
  public L first;
  public R second;
  
  public Pair(L l, R r)
  {
    first = l;
    second = r;
  }
}

/**
  random_index creates non-duplicated random
  integer from [0, range - 1].
  This takes O(N + itr) where itr is the number
  of swapping.
  
  Typical itr is more than 100.
*/
int[] random_index(int range, int size, int itr)
{
  if(range < size)
  {
    throw new IllegalArgumentException("You cannot make " + size + " non-overlapping with " + range + " numbers");
  }
  int[] arr = new int[range];
  for(int i = 0; i < range; ++i)
  {
    arr[i] = i;
  }
  
  // Shuffle
  for(int i = 0; i < itr; ++i)
  {
    // Select non duplicating two positions
    int i1 = (int)(Math.random() * (range - 1));
    int i2 = i1;
    while(i1 == i2)
    {
      i2 = (int)(Math.random() * (range - 1));
    }
    
    // swap
    int temp = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = temp;
  }
  
  // Cut
  int[] out = new int[size];
  System.arraycopy(arr, 0, out, 0, out.length);
  return out;
}

void draw_values(double[] data, double scale, float x, float y, float w, float h)
{
  noStroke();
  for(int i = 0; i < data.length; ++i)
  {
    fill((float)(0.5 + scale * data[i]));
    rect(map(i, 0, data.length, x, x + w), y, w / data.length, h);
  }
}
