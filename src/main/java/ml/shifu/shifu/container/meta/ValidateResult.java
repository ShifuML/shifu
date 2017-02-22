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
package ml.shifu.shifu.container.meta;

import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * ValidateResult class
 */
public class ValidateResult {

    private boolean status;
    private List<String> causes;

    public ValidateResult() {
        this(true);
    }

    public ValidateResult(boolean status) {
        this(status, new ArrayList<String>());
    }

    public ValidateResult(boolean status, List<String> causes) {
        this.status = status;
        this.causes = causes;
    }

    public boolean getStatus() {
        return status;
    }

    public void setStatus(boolean status) {
        this.status = status;
    }

    public List<String> getCauses() {
        return causes;
    }

    public void setCauses(List<String> causes) {
        this.causes = causes;
    }

    public void addCause(String reasonStr) {
        if(causes == null) {
            causes = new ArrayList<String>();
        }

        if(StringUtils.isNotBlank(reasonStr)) {
            status = false;
            causes.add(reasonStr);
        }
    }

    public boolean clearCause() {
        if(causes != null) {
            causes.clear();
        }

        status = true;
        return true;
    }

    @Override
    public String toString() {
        if(status) {
            return "[true]";
        } else {
            return "[false, " + causes.get(0) + "]";
        }
    }

    /**
     * Merge validation results together.
     * The status of total result will be false, if there is one false.
     * The total result will contain all causes
     * 
     * @param resultA
     *            the resultA
     * @param resultB
     *            the resultB
     * @return result after merge
     */
    public static ValidateResult mergeResult(ValidateResult resultA, ValidateResult resultB) {
        ValidateResult finalResult = new ValidateResult(resultA.getStatus() && resultB.getStatus());
        finalResult.getCauses().addAll(resultA.getCauses());
        finalResult.getCauses().addAll(resultB.getCauses());

        return finalResult;
    }

}
