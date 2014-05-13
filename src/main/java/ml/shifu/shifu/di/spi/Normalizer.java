package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.container.obj.ColumnConfig;

public interface Normalizer {

    Double normalize(ColumnConfig config, Object raw);

}
