/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.text.DecimalFormat;

/**
 * JexlTest class
 */
public class JexlTest {

    @Test
    public void testJavaExpressionNum() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "bad_num == 2";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("bad_num", 2);

        // Now evaluate the expression, getting the result
        Boolean isEqual = (Boolean) e.evaluate(jc);
        Assert.assertTrue(isEqual);

        jc.set("bad_num", null);
        isEqual = (Boolean) e.evaluate(jc);
        Assert.assertFalse(isEqual);

        jc.set("bad_num", "2");
        isEqual = (Boolean) e.evaluate(jc);
        Assert.assertTrue(isEqual);
    }

    @Test
    public void testJavaExpressionNotNull() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "bad_num != \"NULL\"";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("bad_num", "2");

        // Now evaluate the expression, getting the result
        Boolean isEqual = (Boolean) e.evaluate(jc);
        Assert.assertTrue(isEqual);

        jc = new MapContext();
        jc.set("bad_num", "NULL");

        // Now evaluate the expression, getting the result
        isEqual = (Boolean) e.evaluate(jc);
        Assert.assertFalse(isEqual);
    }

    @Test
    public void testJavaExpressionString() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "name == \"user_a\"";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("name", "user_a");

        // Now evaluate the expression, getting the result
        Boolean isEqual = (Boolean) e.evaluate(jc);
        Assert.assertTrue(isEqual);

        jc.set("name", "user_b");
        isEqual = (Boolean) e.evaluate(jc);
        Assert.assertFalse(isEqual);
    }

    @Test
    public void testJavaDouble() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "columnA + columnB";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        // Now evaluate the expression, getting the result
        Integer val = (Integer) e.evaluate(jc);
        Assert.assertEquals(val, Integer.valueOf(0));

        jc.set("columnA", "0.3");
        double value = (Double) e.evaluate(jc);
        Assert.assertEquals(0.3, value);

        jc.set("columnB", "0.7");
        value = (Double) e.evaluate(jc);
        Assert.assertEquals(value, 1.0);
    }

    @Test
    public void testJavaNull() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "is_bad_new != null";
        String jexlExpEqual = "is_bad_new == null";

        Expression e = jexl.createExpression(jexlExp);
        Expression exp = jexl.createExpression(jexlExpEqual);

        JexlContext jc = new MapContext();
        jc.set("is_bad_new", null);
        Assert.assertEquals(Boolean.FALSE, e.evaluate(jc));
        Assert.assertEquals(Boolean.TRUE, exp.evaluate(jc));

        jc.set("is_bad_new", new Object());
        Assert.assertEquals(Boolean.TRUE, e.evaluate(jc));
        Assert.assertEquals(Boolean.FALSE, exp.evaluate(jc));
    }

    @Test
    public void testJavaEqual() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "arm14_seg==1 and time_window=='DEV'";

        Expression e = jexl.createExpression(jexlExp);
        JexlContext jc = new MapContext();
        jc.set("arm14_seg", "1");
        jc.set("time_window", "DEV");

        Assert.assertEquals(Boolean.TRUE, e.evaluate(jc));
    }

    @Test
    public void testJavaStrEmpty() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "ARM17_score != null and !ARM17_score.isEmpty()";

        Expression e = jexl.createExpression(jexlExp);
        JexlContext jc = new MapContext();
        jc.set("ARM17_score", null);

        Assert.assertEquals(Boolean.FALSE, e.evaluate(jc));
    }

    @Test
    public void testJavaMode() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "txn_id % 2 == 0 ";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("txn_id", "1");
        Assert.assertEquals(Boolean.FALSE, e.evaluate(jc));

        jc.set("txn_id", "2");
        Assert.assertEquals(Boolean.TRUE, e.evaluate(jc));
        
    }    
    
    @Test
    public void testJavaSubString() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "str.substring(0, 1) == \"a\" ";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("str", "a1");
        Assert.assertEquals(Boolean.TRUE, e.evaluate(jc));
    } 
    
    @Test
    public void testMathMethod() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "NumberUtils.max(a, b, c)";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("NumberUtils", new NumberUtils());
        jc.set("a", 7);
        jc.set("b", 5);
        jc.set("c", 9);
        Assert.assertEquals(9, e.evaluate(jc));
    } 
    
    @Test
    public void testDerived() {
        JexlEngine jexl = new JexlEngine();
        String jexlExp = "(0.00472217*vbase_t1_model_V2R1 + 0.00341543*vbase_t1_model_V2BM)/0.00813760";

        Expression e = jexl.createExpression(jexlExp);

        JexlContext jc = new MapContext();
        jc.set("NumberUtils", new NumberUtils());
        jc.set("vbase_t1_model_V2R1", 238);
        jc.set("vbase_t1_model_V2BM", 289);
        Assert.assertEquals(259.40519686394026, e.evaluate(jc));
    } 
    
    @Test
    public void testDouble() {
        Double a = Double.NaN;
        Double b = Double.valueOf(a.toString());
        Assert.assertEquals(a, b);
    }

    @Test
    public void testEvaluator() {
        String weightColumnName = "cg_dol_wgt";

        JexlEngine jexl = new JexlEngine();
        Expression e = jexl.createExpression(weightColumnName);

        JexlContext jc = new MapContext();
        jc.set("cg_dol_wgt", "1083.22500000");

        Object result = e.evaluate(jc);
        System.out.println((result instanceof Integer));
        System.out.println((result instanceof Double));
        System.out.println(result.toString());
    }

    @Test
    public void testDoubleFormat() {
        Double a = Double.NaN;
        DecimalFormat df = new DecimalFormat("##.######");

        Assert.assertEquals("NaN", a.toString());
        Assert.assertFalse(df.format(a).equals("NaN"));
    }
}
