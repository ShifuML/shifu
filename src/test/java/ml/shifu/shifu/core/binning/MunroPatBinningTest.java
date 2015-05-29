package ml.shifu.shifu.core.binning;

import junit.framework.Assert;
import org.apache.commons.io.IOUtils;
import org.testng.annotations.Test;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by yliu15 on 2014/12/24.
 */
public class MunroPatBinningTest {

    @Test
    public void testBinning() throws IOException {
        MunroPatBinning binning = new MunroPatBinning(10);
        List<String> usageList = IOUtils.readLines(new FileInputStream("src/test/resources/example/binning-data/return_lt_180d_amt.txt"));

        for ( String data : usageList ) {
            binning.addData(data);
        }

        List<Double> binBoundary = binning.getDataBin();
        Assert.assertTrue(binBoundary.size() > 1);
    }

    @Test
    public void tesGussiantBinning() {
        Random rd = new Random(System.currentTimeMillis());

        MunroPatBinning binning = new MunroPatBinning(10);
        for ( int i = 0; i < 100; i ++ ) {
            binning.addData(Double.toString(rd.nextGaussian() % 1000));
        }

        System.out.println(binning.getDataBin());
    }
}
