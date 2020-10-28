import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import ij.plugin.filter.*;
import java.util.Random;

public class Edge_Highlighter implements PlugInFilter {
  int pixel_x;
  int pixel_y;
  int magnitude;
  ImagePlus imp;

  public int setup(String arg, ImagePlus imp){
    this.imp = imp;
    return DOES_RGB+NO_UNDO+NO_CHANGES+DOES_STACKS;
  }


  public void run(ImageProcessor imp) {
    int h = imp.getHeight();
    int w = imp.getWidth();
    ImageProcessor ip = ip.duplicate();
    //byte[] pixels2 = (byte[])imp.getPixels();
  //  int[] pixels = (int[])ip.getPixels();
  //  int red,green,blue;
    for(int x = 1; i< pixels.length; i++){
      for(int y = 1; i< pixels.length; i++){
        pixel_x = 0.5 * (ip.getPixel(x,y+1) - ip.getPixel(x,y) +ip.getPixel(x+1,y+1)-ip.getPixel(x+1,y));
        pixel_y = 0.5 * (ip.getPixel(x+1,y) - ip.getPixel(x,y) +ip.getPixel(x+1,y+1)-ip.getPixel(x,y+1));
        magnitude =  Math.sqrt(pixel_x * pixel_x + pixel_y * pixel_y);
        if (magnitude < 0) {
          magnitude = 0;
        }
        if(magnitude > 255) {
          magnitude = 255;
        }
      }

    }
    ip.putPixel(x,y,magnitude);
  }
}
