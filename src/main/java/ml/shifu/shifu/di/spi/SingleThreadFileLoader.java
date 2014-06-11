package ml.shifu.shifu.di.spi;

import java.util.Map;
import java.util.List;

public interface SingleThreadFileLoader {

    public List<List<Object>> load(String filePath);

}
