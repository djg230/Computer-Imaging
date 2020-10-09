import ij.*;
import ij.process.*;
import ij.gui.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.frame.*;
import java.util.Random;

public class Random_Image implements PlugIn {

	public void run(String arg) {
		int w = 500, h = 500;
		Random r = new Random();
		ImageProcessor ip = new ImageProcessor(w,h);
		int[] pixels = (int[]) ip.getPixels();
		for(int i = 0; i < pixels.length(); i++){
		pixels[i] = r.nextInt(256);
		}
		ip.createImage();
	}

}
