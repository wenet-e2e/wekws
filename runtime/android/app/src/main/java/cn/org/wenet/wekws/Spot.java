package cn.org.wenet.wekws;

public class Spot {

    static {
        System.loadLibrary("wekws");
    }

    public static native void init(String modelDir);
    public static native void reset();
    public static native void acceptWaveform(short[] waveform);
    public static native void setInputFinished();
    public static native void startSpot();
    public static native String getResult();
}
