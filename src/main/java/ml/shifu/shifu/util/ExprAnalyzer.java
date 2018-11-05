package ml.shifu.shifu.util;

import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.List;

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

public class ExprAnalyzer {
    private String expr;

    public ExprAnalyzer(String expr) {
        this.expr = expr;
    }

    public String[] getVarsInExpr() {
        char[] processed = expr.toCharArray();
        removeQuota(processed, '\'');
        removeQuota(processed, '\"');
        removeNonValueChar(processed);
        removeObjMethod(processed);

        String formatStr = new String(processed);
        String[] vars = formatStr.split("[ ]");
        List<String> result = new ArrayList<String>();

        for (String var : vars) {
            String fvar = StringUtils.trimToEmpty(var);
            if (StringUtils.isNotBlank(fvar)
                    && !"and".equals(fvar)
                    && !"or".equals(fvar)
                    && !StringUtils.isNumeric(fvar) ) {
                result.add(fvar);
            }
        }
        return result.toArray(new String[0]);
    }

    private void removeObjMethod(char[] processed) {
        int pos = -1;
        for ( int i = 0; i < processed.length; i ++ ) {
            if (processed[i] == '.') {
                pos = i;
            }

            if ( pos >= 0 && processed[i] == ' ') {
                setBlank(processed, pos, i);
                pos = -1;
            }
        }
    }

    private void removeNonValueChar(char[] processed) {
        for ( int i = 0; i < processed.length; i ++ ) {
            if ( !Character.isLetterOrDigit(processed[i])
                    && processed[i] != '.'
                    && processed[i] != '_' ) {
                processed[i] = ' ';
            }
        }
    }

    private void removeQuota(char[] processed, char c) {
        int startOps = -1;
        int endOps = -1;

        for ( int i = 0; i < processed.length; i ++ ) {
            if ( processed[i] == c ) {
                if ( startOps >= 0 ) {
                    endOps = i;
                } else {
                    startOps = i;
                }

                if ( startOps >= 0 && endOps >= 0 ) {
                    setBlank(processed, startOps, endOps);
                    startOps = -1;
                    endOps = -1;
                }
            }
        }
    }

    private void setBlank(char[] processed, int startOps, int endOps) {
        for ( int i = startOps; i <= endOps; i ++ ) {
            processed[i] = ' ';
        }
    }
}
