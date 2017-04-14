package ml.shifu.shifu.column;

import junit.framework.Assert;
import org.testng.annotations.Test;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by zhanhu on 3/23/17.
 */
public class NSColumnUtilsTest {

    @Test
    public void testNamespaceColumn() {
        Assert.assertTrue(NSColumnUtils.isColumnEqual("diagnosis", "target::diagnosis"));
        Assert.assertFalse(NSColumnUtils.isColumnEqual("other::diagnosis", "target::diagnosis"));
        Assert.assertTrue(NSColumnUtils.isColumnEqual("other::target::diagnosis", "target::diagnosis"));
        Assert.assertTrue(NSColumnUtils.isColumnEqual("", ""));
        Assert.assertFalse(NSColumnUtils.isColumnEqual("", null));
        Assert.assertFalse(NSColumnUtils.isColumnEqual(null, ""));
        Assert.assertTrue(NSColumnUtils.isColumnEqual(null, null));
    }

    @Test
    public void testSetContains() {
        Set<NSColumn> names = new HashSet<NSColumn>();
        names.add(new NSColumn("target::diagnosis"));
        Assert.assertTrue(names.contains(new NSColumn("diagnosis")));

        names = new HashSet<NSColumn>();
        names.add(new NSColumn("cam2015_norm3_LR_5::mean"));
        names.add(new NSColumn("cam2015_norm3_LR_7::mean"));
        names.add(new NSColumn("cam2015_norm3_LR_9::mean"));
        names.add(new NSColumn("cam2015_norm3_NN_0::mean"));
        names.add(new NSColumn("cam2015_norm3_NN_2::mean"));
        names.add(new NSColumn("cam2015_norm3_NN_4::mean"));

        Assert.assertTrue(names.contains(new NSColumn("EvalScore::shifu::cam2015_norm3_NN_0::mean")));
        Assert.assertFalse(names.contains(new NSColumn("EvalScore::shifu::cam2015_norm3_NN_3::mean")));
        Assert.assertFalse(names.contains(new NSColumn("EvalScore::shifu::cam2015_norm3_NN_1::mean")));
    }
}
