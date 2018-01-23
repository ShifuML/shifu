package ml.shifu.shifu.util;

import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import org.junit.Test;
import org.testng.Assert;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by zhanhu on 1/17/18.
 */
public class HdfsPartFileTest {

    @Test
    public void testSingleFile() throws IOException {
        HdfsPartFile hdfsPartFile = new HdfsPartFile(
                "src/test/resources/example/monks-problem/DataStore/DataSet1", SourceType.LOCAL);
        String line = null;
        while ( (line = hdfsPartFile.readLine()) != null) {
            System.out.println(line);
        }

        hdfsPartFile.close();
    }

    @SuppressWarnings("unused")
    @Test
    public void testPartsFile() throws IOException {
        HdfsPartFile hdfsPartFile = new HdfsPartFile(
                "src/test/resources/example/StatsSmallBins", SourceType.LOCAL);
        int lineCnt = 0;
        String line = null;
        while ( (line = hdfsPartFile.readLine()) != null) {
            //System.out.println(line);
            lineCnt++;
        }

        hdfsPartFile.close();
        Assert.assertEquals(lineCnt, 50);
    }

    @Test
    public void testPartsFileMemSize() throws IOException {
        HdfsPartFile hdfsPartFile = new HdfsPartFile(
                "src/test/resources/example/StatsSmallBins", SourceType.LOCAL);
        // String[] smallBinsMap = new String[7930];
        Map<Integer, String> smallBinsMap = new HashMap<Integer, String>();
        int lineCnt = 0;
        String line = null;
        while ( (line = hdfsPartFile.readLine()) != null) {
            String[] fields = line.split("\u0007");
            if ( fields.length == 2 ) {
                // smallBinsMap[Integer.parseInt(fields[0])] = fields[1];
                smallBinsMap.put(Integer.parseInt(fields[0]), fields[1]);
            }
            lineCnt++;
        }

        hdfsPartFile.close();
        Assert.assertEquals(lineCnt, 50);
    }

    @Test
    public void testPartsFileWild() throws IOException {
        String partition = String.format("%05d", 3);
        HdfsPartFile hdfsPartFile = new HdfsPartFile(
                "src/test/resources/example/StatsSmallBins" + File.separator + "part-*-*" + partition + ".*",
                SourceType.LOCAL);
        String line = null;
        int cnt = 0;
        while ( (line = hdfsPartFile.readLine()) != null) {
            System.out.println(line);
            cnt++;
        }

        hdfsPartFile.close();
        Assert.assertEquals(cnt, 10);
    }
}
