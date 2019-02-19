package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.exception.ShifuException;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

public class AssertUtilsTest {

    @Test
    public void testAssertEqual() {
        AssertUtils.assertEquals(1.0f, 1.0f);
        AssertUtils.assertEquals(1.0, 1.0d);
        AssertUtils.assertEquals(1, 1);
        AssertUtils.assertEquals("hello", "hello");
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testAssertFloatFail() {
        AssertUtils.assertEquals(1.0f, 1.1f);
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testAssertDoubleFail() {
        AssertUtils.assertEquals(1.0d, 1.1d);
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testAssertStringFail() {
        AssertUtils.assertEquals("shifu", "master");
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testAssertIntegerFail() {
        AssertUtils.assertEquals(1, 2);
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testListSizeNotEqual() {
        List<Float> a = new ArrayList<>();
        a.add(3f);
        List<Float> b = new ArrayList<>();
        AssertUtils.assertListNotNullAndSizeEqual(a, b);
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testListNullNotEqual() {
        List<Float> a = new ArrayList<>();
        a.add(3f);
        AssertUtils.assertListNotNullAndSizeEqual(a, null);
    }

    @Test
    public void testListSizeEqual() {
        List<Float> a = new ArrayList<>();
        a.add(3f);
        List<Float> b = new ArrayList<>();
        b.add(2f);
        AssertUtils.assertListNotNullAndSizeEqual(a, b);
    }

    @Test(expectedExceptions = { ShifuException.class })
    public void testArrayNullNotEqual() {
        float[] a = { 3f };
        AssertUtils.assertFloatArrayNotNullAndLengthEqual(a, null);
    }

    @Test
    public void testArrayLengthEqual() {
        float[] a = { 3f };
        float[] b = { 5f };
        AssertUtils.assertFloatArrayNotNullAndLengthEqual(a, b);
    }
}
