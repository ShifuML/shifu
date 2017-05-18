package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.junit.Test;
import org.testng.Assert;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by zhanhu on 4/18/17.
 */
public class DynamicCategoricalBinTest {

    @Test
    public void testDynamicCategoricalBin() throws IOException {
        List<AbstractBinInfo> categoricalBinInfoList = loadTestData();
        Collections.sort(categoricalBinInfoList);

        AutoDynamicBinning binning = new AutoDynamicBinning(2);
        List<AbstractBinInfo> finalBins = binning.merge(categoricalBinInfoList);

        Assert.assertEquals(finalBins.size(), 2);

        List<String> categoricalVals = new ArrayList<String>();
        List<Long> negativeCnts = new ArrayList<Long>();
        List<Long> positiveCnts = new ArrayList<Long>();
        List<Double> positiveRates = new ArrayList<Double>();

        for ( int i = 0; i < finalBins.size(); i ++ ) {
            CategoricalBinInfo binInfo = (CategoricalBinInfo)finalBins.get(i);
            categoricalVals.add("\"" + StringUtils.join(binInfo.getValues(), '^') + "\"");
            negativeCnts.add(binInfo.getNegativeCnt());
            positiveCnts.add(binInfo.getPositiveCnt());
            positiveRates.add(binInfo.getPositiveRate());
        }

        System.out.println(StringUtils.join(categoricalVals, ','));
        System.out.println(StringUtils.join(negativeCnts, ','));
        System.out.println(StringUtils.join(positiveCnts, ','));
        System.out.println(StringUtils.join(positiveRates, ','));
    }

    private List<AbstractBinInfo> loadTestData() throws IOException {
        List<String> lines = IOUtils.readLines(DynamicCategoricalBinTest.class
                .getResourceAsStream("/example/binning-data/categorical-binning"));

        String[] categories = lines.get(0)
                .replaceAll("^.* \\[", "")
                .replaceAll("].*$", "")
                .replaceAll("\"", "").trim().split(",");

        String[] binPosCounts = lines.get(2)
                .replaceAll("^.* \\[", "")
                .replaceAll("].*$", "")
                .replaceAll("\"", "").trim().split(",");

        String[] binNegCounts = lines.get(1)
                .replaceAll("^.* \\[", "")
                .replaceAll("].*$", "")
                .replaceAll("\"", "").trim().split(",");

        @SuppressWarnings("unused")
        String[] positiveRates = lines.get(3)
                .replaceAll("^.* \\[", "")
                .replaceAll("].*$", "")
                .replaceAll("\"", "").trim().split(",");

        List<AbstractBinInfo> categoricalBinInfos = new ArrayList<AbstractBinInfo>();
        for ( int i = 0; i < categories.length; i ++ ) {
            CategoricalBinInfo binInfo = new CategoricalBinInfo();
            List<String> values = new ArrayList<String>();
            values.add(categories[i].trim());
            binInfo.setValues(values);

            binInfo.setPositiveCnt(Long.parseLong(binPosCounts[i].trim()));
            binInfo.setNegativeCnt(Long.parseLong(binNegCounts[i].trim()));

            categoricalBinInfos.add(binInfo);
        }

        return categoricalBinInfos;
    }
}
