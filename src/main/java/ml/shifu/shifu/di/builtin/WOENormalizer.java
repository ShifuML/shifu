package ml.shifu.shifu.di.builtin;

import ml.shifu.shifu.di.spi.Normalizer;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.CommonUtils;

public class WOENormalizer implements Normalizer {

     public Double normalize(ColumnConfig config, Object raw) {

         int binNum = CommonUtils.getBinNum(config, raw);
         return config.getColumnBinStatsResult().getBinWoe().get(binNum);
     }
}
