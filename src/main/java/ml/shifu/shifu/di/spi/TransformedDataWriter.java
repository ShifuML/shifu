package ml.shifu.shifu.di.spi;


import java.util.List;

public interface TransformedDataWriter {
    public void println(List<Object> row);
}
