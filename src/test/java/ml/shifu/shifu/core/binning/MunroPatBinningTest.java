/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.binning;

import junit.framework.Assert;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.io.IOUtils;
import org.testng.annotations.Test;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

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

    @Test
    public void testSmallBinning() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig("src/test/resources/example/inner_seg1_v15/ModelConfig.json", RawSourceData.SourceType.LOCAL);

        MunroPatBinning inst = new MunroPatBinning(2000, modelConfig.getMissingOrInvalidValues());
        InputStream is = MunroPatBinningTest.class.getResourceAsStream("/example/inner_seg1_v15/dib_sample_fields.gz");
        GZIPInputStream gzis = new GZIPInputStream(is);

        List<String> lines = IOUtils.readLines(gzis);
        for (String record : lines) {
            String[] fields = record.split("\\|");
            inst.addData(fields[0]);
        }

        List<Double> boundaries = inst.getDataBin();
        System.out.println(boundaries);
        Assert.assertTrue(boundaries.size() > 1000);

        IOUtils.closeQuietly(gzis);
        IOUtils.closeQuietly(is);
    }
}
