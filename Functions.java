// 2018-07-27: add ABS

import java.util.*;
interface IFunction
{
  public double f(double x);
  public double dfdx(double x, double y);
  public String getName();
}

public class Functions
{
  public static void init()
  {
    //Functions.FUNC_MAP.put("SIGMOID", Functions.SIGMOID);
    Functions.FUNC_MAP.put("RELU", Functions.RELU);
    Functions.FUNC_MAP.put("LEAKY_RELU", Functions.LRELU);
    Functions.FUNC_MAP.put("ISRLU", Functions.ISRLU);
    Functions.FUNC_MAP.put("ELU", Functions.ELU);
    Functions.FUNC_MAP.put("TANH", Functions.TANH);
    Functions.FUNC_MAP.put("TANH_01", Functions.TANH_01);
    Functions.FUNC_MAP.put("GAUSSIAN", Functions.GAUSSIAN);
  }
  
  public static Map <String, IFunction> FUNC_MAP = new HashMap <String, IFunction> ();
  
  public static IFunction SIGMOID = new IFunction() {
    public double f(double x) {
      return 1.0/(1.0 + Math.exp(-x));
    }
    public double dfdx(double x, double y) {
      return y * (1 - y);
    }
    public String getName() { return "SIGMOID"; }
  };
  
  /**
    Hyperbolic Tangent.
    Range = (-1, 1)
    Bigger gradient than sigmoid.
  */
  public static IFunction TANH = new IFunction() {
    public double f(double x) {
      double temp = Math.exp(x);
      return (temp - 1/temp) / (temp + 1/temp);
    }
    public double dfdx(double x, double y) {
      return 1 - y * y;
    }
    public String getName() { return "TANH"; };
  };
  
  /**
    New generation of bounded activation function.
    Computation is more easier than TanH
    Actually this is worthy only when it is implemented
    with C/C++.
    Range = (-1, 1)
  */
  public static IFunction ISRU = new IFunction() {
    private double a = 1;
    public double f(double x) {
      return x / Math.sqrt(1 + a*x*x);
    }
    public double dfdx(double x, double y) {
      return (x == 0 ? 1 : y/x * y/x * y/x);
    }
    public String getName() { return "ISRU"; }
  };
  
  /**
    Very famous activation function.
    Range = [0, inf)
  */
  public static IFunction RELU = new IFunction() {
    public double f(double x) {
      return (x >= 0 ? x : 0);
    }
    public double dfdx(double x, double y) {
      return (x >= 0 ? 1 : 0);
    }
    public String getName() { return "RELU"; }
  };
  
  /**
    More robust version of ReLU
    Range = (-inf, inf)
  */
  public static IFunction LRELU = new IFunction() {
    public double f(double x) {
      return (x >= 0 ? x : 0.1 * x);
    }
    public double dfdx(double x, double y) {
      return (x >= 0 ? 1 : 0.1);
    }
    public String getName() { return "LRELU"; }
  };
  
  /**
    [-1, inf)
  */
  public static IFunction ELU = new IFunction() {
    public double f(double x) {
      return (x >= 0 ? x : 0.05 * (Math.exp(x)-1));
    }
    public double dfdx(double x, double y) {
      return (x >= 0 ? 1 : y + 0.05);
    }
    public String getName() { return "ELU"; }
  };
  
  /**
    Similar to ELU but it converges more faster.
    Range = (-1, inf)
  */
  public static IFunction ISRLU = new IFunction() {
    private double a = 1;
    public double f(double x) {
      return (x >= 0 ? x : x / Math.sqrt(1 + a * x * x));
    }
    public double dfdx(double x, double y) {
      return (x == 0 ? 1 : y/x * y/x * y/x);
    }
    public String getName() { return "ISRLU"; }
  };
  
  /**
    Hyperbolic Tangent with adjusted bias.
    Range = (0, 1)
    Compatible with sigmoid. It's more powerful.
  */
  public static IFunction TANH_01 = new IFunction() {
    public double f(double x) {
      return 0.5 + 0.5 * TANH.f(x);
    }
    public double dfdx(double x, double y) {
      return 0.5 * TANH.dfdx(x, y);
    }
    public String getName() { return "TANH_01"; }
  };
  
  /**
    Simple swish function proposed by Google.
    IFunction version doesn't support trainable beta.
    If you want to do that, you should add swish module.
    Range = [-0.2784.., inf)
  */
  public static IFunction SWISH = new IFunction() {
    private double sigmoid(double x) {
      return 1.0 / (1.0 + Math.exp(-x));
    }
    public double f(double x) {
      return x * sigmoid(x);
    }
    public double dfdx(double x, double y) {
      return y + sigmoid(x) * (1 - y);
    }
    public String getName() { return "SWISH"; }
  };
  
  public static IFunction ABS = new IFunction() {
    public double f(double x) {
      return Math.abs(x);
    }
    public double dfdx(double x, double y) {
      return x > 0 ? 1 : (x < 0 ? -1 : 0);
    }
    public String getName() { return "ABS"; }
  };
  
  public static IFunction GAUSSIAN = new IFunction() {
    public double f(double x) {
      return Math.exp(-x * x);
    }
    public double dfdx(double x, double y) {
      return -2 * x * y;
    }
    public String getName() { return "GAUSSIAN"; }
  };
  
  public static IFunction SQUARE = new IFunction() {
    public double f(double x) {
      return x * x;
    }
    public double dfdx(double x, double y) {
      return 2 * x;
    }
    public String getName() { return "SQUARE"; }
  };
  
  public static IFunction SQRT = new IFunction() {
    public double f(double x) {
      return Math.sqrt(x);
    }
    public double dfdx(double x, double y) {
      return 0.5 / y;
    }
    public String getName() { return "SQRT"; }
  };
  
  public static IFunction LOG = new IFunction() {
    public double f(double x) {
      return Math.log(x);
    }
    public double dfdx(double x, double y) {
      return 1.0 / x;
    }
    public String getName() { return "LOG"; }
  };
  
  public static IFunction EXP = new IFunction() {
    public double f(double x) {
      return Math.exp(x);
    }
    public double dfdx(double x, double y) {
      return y;
    }
    public String getName() { return "EXP"; }
  };
}
