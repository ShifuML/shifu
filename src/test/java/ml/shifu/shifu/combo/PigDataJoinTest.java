package ml.shifu.shifu.combo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import junit.framework.Assert;

import org.apache.commons.lang.StringUtils;
import org.testng.annotations.Test;

/**
 * Created by zhanhu on 12/13/16.
 */
public class PigDataJoinTest {

    @Test
    public void testGenPigJoinCode() throws IOException {
        PigDataJoin inst = new PigDataJoin();

        List<ColumnFile> columnFileList = new ArrayList<ColumnFile>();

        HashMap<String, String> varsMapping1 = new HashMap<String, String>();
        varsMapping1.put("mean", "model0");
        columnFileList.add(new ColumnFile("Model_NN_0/evals/EvalA", ColumnFile.FileType.PIGSTORAGE, "|",
                new String[]{"id", "mean"}, varsMapping1));

        HashMap<String, String> varsMapping2 = new HashMap<String, String>();
        varsMapping2.put("mean", "model1");
        columnFileList.add(new ColumnFile("Model_NN_1/evals/EvalA", ColumnFile.FileType.PIGSTORAGE, "|",
                new String[]{"mean"}, varsMapping2));

        String pigCode = inst.genPigJoinCode("id", "tmp/Eval1AssembleData", columnFileList);
        System.out.println(pigCode);
        Assert.assertTrue(StringUtils.isNotBlank(pigCode));
    }
}
