/**
  Spectrogram class stores the frequency spectrogram
  on the 2D array, and provide O(1) interface to
  write, read in time-frequency domain.
  
  This class doesn't shift vectors of each times.
  It works more likely with circular array list.
  
  Note that first spectrum is filled with zero.
*/
public class Spectrogram
{
  // [time][freq]
  private float[][] data;
  
  // point the current head position
  private int ptr;
  
  /**
    Create the spectrogram having its time capacity
    as time_length, frequency capacity as freq_bins.
  */
  public Spectrogram(int time_length, int freq_bins)
  {
    data = new float[time_length][freq_bins];
    ptr = 0;
  }
  
  /**
    Return the spectrogram's time capacity
  */
  public int time_length()
  {
    return data.length;
  }
  
  /**
    Return the spectrogram's frequency capacity
  */
  public int freq_bins()
  {
    return data[0].length;
  }
  

  /**
    Write the one spectrum frame to this queue.
    If it reaches the limit, it overwrites the
    oldest one.
  */
  public synchronized void write(float[] spectrum)
  {
    ptr = (ptr + 1) % time_length();
    System.arraycopy(spectrum, 0, data[ptr], 0, freq_bins());
  }
  
  /**
    Return the original spectrum array at the given
    relative index. offset can be any arbitrary integer.
  */
  public synchronized float[] read(int offset)
  {
    return data[index_of(offset)];
  }
  
  /**
    Return the physical index with given relative
    index: offset. offset can be any arbitrary
    integer.
  */
  protected int index_of(int offset)
  {
    return ((ptr + offset) % time_length() + time_length()) % time_length();
  }
}
