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
package ml.shifu.shifu.exception;

/**
 * ShifuException, contain error code
 */
public class ShifuException extends RuntimeException {

    /**
     * serialVersionUID
     */
    private static final long serialVersionUID = 320754504510306120L;
    /**
     * error code
     */
    private ShifuErrorCode error = null;

    public ShifuException(ShifuErrorCode code) {
        super();
        setError(code);
    }

    public ShifuException(ShifuErrorCode code, Exception e) {
        super(e);
        this.setError(code);
    }

    public ShifuException(ShifuErrorCode code, String msg) {
        super(msg);
        this.setError(code);
    }

    public ShifuException(ShifuErrorCode code, Exception e, String msg) {
        super(msg, e);
        this.setError(code);
    }

    public ShifuErrorCode getError() {
        return error;
    }

    public void setError(ShifuErrorCode error) {
        this.error = error;
    }

    @Override
    public String toString() {
        return "ShifuException [error=" + error + "]";
    }

}
