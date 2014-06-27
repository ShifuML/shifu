package ml.shifu.core.di.spi;


import java.util.List;

public interface TransformedDataWriter {
    public void println(List<Object> row);
}
