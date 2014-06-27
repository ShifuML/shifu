package ml.shifu.core.di.spi;

import java.util.List;

public interface SingleThreadFileLoader {

    public List<List<Object>> load(String filePath);

}
