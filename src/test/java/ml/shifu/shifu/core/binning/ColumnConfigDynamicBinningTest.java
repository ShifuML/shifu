package ml.shifu.shifu.core.binning;


import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.util.CommonUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.List;

/**
 * Created by zhanhu on 5/8/17.
 */
public class ColumnConfigDynamicBinningTest {

    @Test
    public void testCategoricalConfigDynamicBinning() throws IOException {
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/binning-data/ColumnConfig.json", RawSourceData.SourceType.LOCAL);
        ColumnConfigDynamicBinning ccdb = new ColumnConfigDynamicBinning(
                columnConfigList.get(0), 0, 0.99, 0);
        List<AbstractBinInfo> binInfos= ccdb.run();
        Assert.assertEquals(binInfos.size(), 11);
    }

    @Test
    public void testNumericalConfigDynamicBinning() throws IOException {
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/binning-data/ColumnConfig.json", RawSourceData.SourceType.LOCAL);
        ColumnConfigDynamicBinning ccdb = new ColumnConfigDynamicBinning(
                columnConfigList.get(2), 0, 0.99, 2000);
        List<AbstractBinInfo> binInfos= ccdb.run();
        Assert.assertEquals(binInfos.size(), 41);
    }

    @Test
    public void testNumericalConfigDynamicBinning2() throws IOException {
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/example/binning-data/ColumnConfig.json", RawSourceData.SourceType.LOCAL);
        ColumnConfigDynamicBinning ccdb = new ColumnConfigDynamicBinning(
                columnConfigList.get(3), 0, 0.99, 2000);
        List<AbstractBinInfo> binInfos= ccdb.run();
        Assert.assertEquals(binInfos.size(), 17);
    }

    @Test
    public void testIvReduce() {
        double iv1 = calcualte(0, 28);
        double iv2 = calcualte(0, 30);
        double iv = calcualte(0, 58);
        Assert.assertTrue(iv1 + iv2 < iv);
    }

    public double calcualte(double cntP, double cntN) {
        double p = cntP / 1212895.0;
        double n = cntN / 1.1486916E7;

        double woePerBin = Math.log((p + EPS) / (n + EPS));
        return (p - n) * woePerBin;
    }

    private final static double EPS = 1e-10;
}
