package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.core.binning.obj.NumBinInfo;
import org.apache.commons.lang.StringUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningTest {

    @Test
    public void testDIB() {
        List<NumBinInfo> binInfoList = createNumBinInfos(30);
        DynamicBinning dynamicBinning = new DynamicBinning(binInfoList, 10);
        Assert.assertEquals(dynamicBinning.getDataBin().size(), 10);
    }

    public static List<NumBinInfo> createNumBinInfos(int binCnt) {
        Random rd = new Random(System.currentTimeMillis());

        List<Double> thresholds = new ArrayList<Double>(binCnt - 1);
        for ( int i = 0; i < binCnt - 1; i ++ ) {
            thresholds.add(rd.nextGaussian() * 200);
        }

        Collections.sort(thresholds);

        List<NumBinInfo> binInfoList = NumBinInfo.constructNumBinfo(StringUtils.join(thresholds, ':'), ':');
        for ( NumBinInfo binInfo : binInfoList ) {
            if ( rd.nextDouble() > 0.45 ) {
                int total = Math.abs(rd.nextInt()) % 1000;
                int positive = (int) (total * rd.nextDouble());
                binInfo.setTotalInstCnt(total);
                binInfo.setPositiveInstCnt(positive);
            }
        }

        return binInfoList;
    }

}
