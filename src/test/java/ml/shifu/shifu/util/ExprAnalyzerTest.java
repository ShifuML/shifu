package ml.shifu.shifu.util;

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ExprAnalyzerTest {

    @Test
    public void testExprAnalyzer() {
        ExprAnalyzer analyzer = new ExprAnalyzer("(pop_type=='ach' and cnsr_seg=='Y') or pop_type=='cc_'");
        Assert.assertEquals(analyzer.getVarsInExpr()[0], "pop_type");
        Assert.assertEquals(analyzer.getVarsInExpr()[1], "cnsr_seg");
        Assert.assertEquals(analyzer.getVarsInExpr()[2], "pop_type");
    }

    @Test
    public void testExprAnalyzer2() {
        ExprAnalyzer analyzer = new ExprAnalyzer("auth_capture_flag.matches('auth|sale') and robust_dev_flag==1");
        Assert.assertEquals(analyzer.getVarsInExpr()[0], "auth_capture_flag");
        Assert.assertEquals(analyzer.getVarsInExpr()[1], "robust_dev_flag");
    }
}
