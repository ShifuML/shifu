/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.util;

import org.apache.commons.jexl2.Expression;
import org.apache.commons.jexl2.JexlContext;
import org.apache.commons.jexl2.JexlEngine;
import org.apache.commons.jexl2.MapContext;
import org.testng.Assert;
import org.testng.annotations.Test;

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
}
