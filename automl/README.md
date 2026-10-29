# Fraud detection with AutoML

This is an additional, standalone workshop flow: **explore → submit a managed AutoML experiment → inspect results → deploy → score Sally's transaction**. The existing Keras/ONNX, Ray, Training Operator and Feast chapters remain available.

Use the seven feature columns in the repository CSV files, with `fraud` as the binary label. The original `train.csv` and `validate.csv` are merged by default; `test.csv` is supplied through the managed pipeline's external test inputs. No scaling, feature filtering or class resampling is done by this sample. AutoML handles preprocessing.

## Before opening the notebooks

1. Use a RHOAI project with a running Data Science Pipelines server, the GA `autogluon-tabular-training-pipeline` registered as a managed pipeline, and model serving enabled. The installed pipeline must support `preset`, `eval_metric`, `positive_class` and `test_data_*`. Ask your administrator to enable/update these components if the preflight check fails. This sample does not install cluster operators or upload a replacement pipeline.
2. Create an **S3-compatible Connection** in that project. Record its Kubernetes **Secret name** (not the display name). It needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION`, and `AWS_S3_BUCKET`. The pipeline uses this Secret to read both training and external test data; the workbench uses the matching environment variables. Grant read/write/list access to the input and artifact prefixes. The bucket must already exist and its endpoint must be reachable from the workbench, pipeline pods and serving pods.
3. Attach the Connection to a Python 3.11+ workbench and clone this repository. Use the project's pipeline artifact bucket for this Connection when possible. Pipeline **output storage** is configured on the pipeline server, independently of `train_data_secret_name`: this sample does not change that server configuration. Set `S3_ARTIFACTS_BUCKET` to the server's actual artifact bucket. If it differs from the input bucket, the workbench credentials need access to both; create a second Connection for the artifact bucket and use **dashboard deployment (Option A)**. The current programmatic deployment helper (Option B) reuses the Connection recorded during staging and requires that Connection's `AWS_S3_BUCKET` to equal the artifact bucket. Access to both buckets alone does not satisfy that check.
4. Copy [`.env.example`](.env.example) to `.env` **inside `automl/`** and fill in the project namespace, pipeline API endpoint, token and Connection values. `RHOAI_TOKEN` can be obtained with `oc whoami -t`. For inference, leave `PREDICT_TOKEN` empty and use the default `PREDICT_AUTH_MODE=auto`: each prediction reads the current kubeconfig identity (the context used by `oc login`) or the workbench ServiceAccount. That identity must have model inference access. An explicit `PREDICT_TOKEN` overrides automatic credentials. Do not paste tokens into notebook cells. Existing environment values, including attached Connection variables, take precedence over `.env`; see [Changing configuration and reusing data](#changing-configuration-and-reusing-data) for how file edits interact with clients and saved state.
5. For a private certificate authority, configure `REQUESTS_CA_BUNDLE` for pipeline/inference HTTPS and `AWS_CA_BUNDLE` for S3. Keep certificate verification enabled.

Start with [00_prepare_data.ipynb](00_prepare_data.ipynb). Its first cell installs the notebook dependencies; restart the kernel if prompted. The same environment can also be prepared with `python -m pip install -r automl/requirements.txt` from the repository root. Local execution additionally needs network access to the cluster and S3, plus `oc login`/kubeconfig for programmatic deployment. A workbench uses its ServiceAccount permissions for Kubernetes operations; the pipeline API token does not grant those permissions.

## Notebook sequence

| Notebook | What it does | Saved evidence |
| --- | --- | --- |
| [00 — Prepare data](00_prepare_data.ipynb) | Explore class imbalance, validate all columns, optionally merge validation data, upload unique input copies to S3 | Data manifest, row counts, class rates, SHA-256 hashes, S3 URIs |
| [01 — Experiment](01_experiment.ipynb) | Resolve the installed managed pipeline/version, check inputs, submit and monitor | Run ID, pipeline version, actual arguments, status and duration |
| [02 — Results](02_results.ipynb) | Validate external test artifact, rank predictors by the configured metric (`roc_auc`, `f1` or `accuracy`), show leaderboard, retrieve official experiment and predictor notebooks | Holdout scores, selected predictor URI, artifact index, generated notebook links |
| [03 — Predictor](03_predictor.ipynb) | Deploy using the installed `kserve-autogluon-server` runtime and an existing Connection; send REST requests with seven named features | Deployment manifests, prediction response, ticket comment |

Every notebook has previous/next links. Each loads saved state from `automl/outputs/`, so it can use a separate kernel. `.env` and the `outputs/` directory (including downloaded generated notebooks and model artifacts) are ignored by Git. Outputs embedded in the four tracked workshop notebooks are **not** excluded by `.gitignore`. The shared [workflow.py](workflow.py) contains S3, KFP and serving plumbing; notebook cells show the experiment decisions and results.

`speed` is the default GA preset; use `balanced` for a later comparison if resources and time allow. These names are managed pipeline presets, not `TabularPredictor.fit` SDK presets. The referenced pipeline allocates about 4 CPUs/16 GiB for speed selection and 8 CPUs/32 GiB for balanced selection; its steps can have higher limits. Plan for workspace PVC storage (currently 32 GiB in the managed definition), model artifacts, refitting, evaluation and scheduling. Confirm quotas against your installed pipeline. Speed's selection budget is 45 minutes and balanced's is 180 minutes, so total run time can be longer.

The submission cell uses `START_NEW_RUN = False`: it submits the first run when `outputs/run.json` does not exist, then returns that saved run on subsequent executions, including when the saved run has failed. Set `START_NEW_RUN = True` explicitly to submit another experiment, then set it back to `False`. Each execution with `True` submits another run. Before a new experiment, copy `outputs/` to a backup outside the repository to preserve the previous evidence. Keep `outputs/data_manifest.json` in its working location so the existing S3 data can be reused. A monitoring timeout or interrupted kernel does not cancel the pipeline. Resume monitoring using the recorded run ID. Cancel unwanted runs through the Pipelines UI.

Set `RHOAI_DASHBOARD_URL` to the RHOAI dashboard base URL to display clickable **Run details** and **AutoML results** links for either a new or saved run. `KFP_ENDPOINT` remains the pipeline API address. The SDK's default Kubeflow UI links are suppressed during submission because their `/#/runs/details/...` paths do not point at the RHOAI dashboard. Link paths use the saved run's project namespace and ID.

## Changing configuration and reusing data

`wf.settings()` rereads `.env` on each call without copying its values into the kernel environment. File changes take effect when a helper next reads configuration, unless a workbench/kernel environment variable overrides that value. Restart the kernel after changing inherited variables or values previously loaded with `load_dotenv(..., override=True)`. After an edit, rerun the setup cells and recreate existing clients (`wf.s3_client()` / `wf.kfp_client()`); an already-created client retains its earlier configuration. If `workflow.py` itself changed, restart the kernel or reload it with `importlib.reload(wf)`.

Saved experiment state is separate from the live configuration:

- `outputs/data_manifest.json` records the uploaded CSV locations, preparation settings, statistics and the Connection name used at staging time.
- `outputs/run.json` records the submitted arguments, namespace, pipeline version and artifact location. Editing `.env` does not update an existing run or replace this snapshot.
- `outputs/results.json` and `outputs/serving_result.json` describe the run whose ID they contain. Complete notebook 02 for a new run before deploying or scoring it.

| Change | What to do | Upload CSVs again? |
| --- | --- | --- |
| Correct `S3_CONNECTION_NAME` | Update the local manifest as shown below; submit a new experiment using `START_NEW_RUN = True` | No, if the new Connection can read the existing S3 paths |
| Change preset, metric, top N, pipeline version or experiment name | Update `.env`; submit a new experiment explicitly | No |
| Change dataset contents, `MERGE_VALIDATION`, or the desired input bucket/prefix | Update configuration and rerun notebook 00's upload cell; reload `manifest` in notebook 01 and submit a new experiment | Yes; editing configuration does not move existing objects |
| Correct the artifact bucket/prefix for an existing run | Verify its actual output URIs, correct only `artifact_bucket` / `artifact_prefix` in `run.json`, then rerun notebook 02 | No; `.env` does not change pipeline server output storage |
| Change prediction endpoint/token or authentication mode | Update `.env` and rerun the scoring cell; renew `oc login` when needed | No |
| Change model/runtime name or serving resources | Regenerate manifests, then deploy; existing resources with different requested fields must be reviewed or given new names | No |

For a corrected training Connection, run this after editing `.env` and refreshing configuration. This updates only local staging metadata; it does not upload CSVs or modify the previous run:

```python
manifest = wf.read("data_manifest.json")
manifest["connection"] = wf.required("S3_CONNECTION_NAME")
wf.save("data_manifest.json", manifest)
```

Then run notebook 01's submission cell once with `START_NEW_RUN = True`, set it back to `False`, and monitor the new run. Do not edit the previous run's arguments to make it appear to have used the corrected Connection. When switching projects, ensure the Secret exists in the new namespace and make sure your pipeline API endpoint, Kubernetes context and project agree before submitting.

## Interpreting the results

Set `AUTOML_EVAL_METRIC` to `roc_auc`, `f1` or `accuracy`. ROC-AUC is the default ranking metric because accuracy can conceal missed fraud in this imbalanced dataset. The notebook also shows available F1, precision and recall metrics. The pipeline evaluates and ranks candidates on the external `test.csv`; consequently this is an **external evaluation set used in model selection**, not an untouched final generalization estimate. Do not tune repeatedly against it and then describe its score as independent validation. A separate final dataset would be needed for that claim.

Notebook 02 downloads the **official run-level experiment notebook** and the **official selected predictor notebook** from the managed pipeline's artifacts. Open the displayed links to inspect/reproduce the experiment or perform local `predict_proba` with the generated predictor notebook. Those notebooks carry the installed component's dependency/version instructions: follow them in a separate kernel/workbench if necessary. Use the same Connection credentials, with `AWS_S3_BUCKET` pointing to the artifact bucket. Loading a serialized predictor requires trusting its source; use artifacts from your own experiment. Notebook 03 uses REST inference and does not install AutoGluon in the workshop kernel.

## Serving and troubleshooting

Notebook 03 documents dashboard deployment and provides optional cells to create a ServingRuntime from the installed RHOAI template plus an InferenceService. Use one deployment method. Programmatic creation requires read access to that template/Connection Secret and read/create access to ServingRuntimes, InferenceServices, ServiceAccounts, Roles and RoleBindings in your project. The generated storage ServiceAccount receives only `get` access to the selected Connection Secret. Repeating creation reuses resources whose requested fields match, creates missing resources, and reports `created`/`reused`. A configuration mismatch stops with differing field names; existing resources are never overwritten. Authentication remains enabled; configure model access for your inference identity in the dashboard. The service reads the **whole selected `predictor/` directory**, not a single `.pkl` file. The runtime version must match the managed training components.

For Option B, the Connection stored in `run.json` must point at the artifact bucket. Changing `S3_CONNECTION_NAME` after training does not select a different serving Connection for that run. If input and artifact buckets differ, use Option A with a separate artifact-bucket Connection; a second Connection is not selected automatically by the programmatic helper.

- No managed pipeline or missing inputs: ask the administrator to check the project's managed pipeline registration/version.
- Runtime template not found (HTTP 404): run `oc get templates -n redhat-ods-applications` (use your configured runtime namespace if different). Set `RUNTIME_TEMPLATE_NAME` to the installed AutoGluon template's `metadata.name`; the template name can differ from the `kserve-autogluon-server` image name. Manifest generation reports available templates and AutoGluon candidates when listing is permitted. If none is installed, ask the administrator to enable a compatible AutoGluon serving runtime. This does not require another training run or data upload.
- Endpoint still shows `YOUR_REGION` after editing `.env`: check that the edited file is `automl/.env`. The updated loader reads file changes on each call without caching them in the kernel environment. If an earlier version already loaded the placeholder, restart the kernel once. Workbench environment variables still override the file; correct any placeholder in the attached Connection/workbench configuration as well.
- No artifacts: inspect the successful run's output URIs in Pipelines. Check `S3_ARTIFACTS_BUCKET` and the artifact prefix convention. For an existing run update only `artifact_bucket`/`artifact_prefix` in `outputs/run.json`; rerun notebook 02. Do not start training again just to locate artifacts.
- Failed/OOM: inspect failed task logs and pod termination reasons; record `OOMKilled` only if observed. Check quota and workspace storage before trying again.
- Predictor not ready: inspect model events, storage initializer logs, Connection endpoint/bucket, runtime version and memory. An AlreadyExists response is handled by checking and reusing matching resources, so rerunning the creation cell can complete a partial deployment. If requested fields differ, review the reported fields in the dashboard or choose another MODEL_NAME/SERVING_RUNTIME_NAME and regenerate the manifests. No existing resources are patched or deleted.
- HTTP 401/403: in auto mode renew `oc login` or check the workbench identity and its model inference permissions. A nonempty `PREDICT_TOKEN` overrides automatic credentials; clear any stale override from both `.env` and the kernel environment. Credentials are read per prediction without being written to disk; static expired tokens still require renewing the login. The helper does not mint tokens, grant model access or disable deployment authentication. For an intentionally public endpoint, use `PREDICT_AUTH_MODE=none` to omit Authorization; `token` mode requires an explicit `PREDICT_TOKEN`. HTTP 404: verify the base endpoint and that `MODEL_NAME` matches the deployed service. The REST contract is KServe v1 `instances` containing dictionaries with one-element column lists (for example, `"used_chip": [1.0]`). The helper converts the notebook's scalar records into that runtime format; the original five-feature ONNX tensor is not used.
- HTTP 500: inspect the server response included in the exception and the `kserve-container` logs, for example `oc logs -n YOUR_PROJECT -l serving.kserve.io/inferenceservice=fraud-automl -c kserve-container --tail=100`. The helper truncates response details and redacts credentials known to the request. Check the actual traceback before changing request format, runtime version or model artifacts; HTTP 500 alone does not establish the cause. A Ready service can still fail a prediction.
- FastAI error `cannot access local variable 'res' where it is not associated with a value`: this can mask an import failure while loading an exported FastAI learner. In the observed environment, the artifact referenced `ray.experimental.tqdm_ray.safe_print`, but the serving image had no Ray package; direct loading exposed `ModuleNotFoundError: No module named 'ray'`. Training recorded Ray 2.58.0 and FastAI 2.8.12. This requires a compatible serving image or a corrected export that removes the training-only Ray reference. Installing Ray in the notebook does not change the serving container. Stacked predictors can depend on FastAI even when the selected model is named RandomForest or LightGBM. Preserve the failed smoke-test result and report the artifact/runtime compatibility problem to the AutoML maintainers.

The last notebook writes `outputs/ticket-comment.md` with observed S3 paths, dataset statistics, run details, metric and serving result. Fill in branch/PR links and failure/OOM evidence before copying it to the ticket. No score or serving success is recorded until measured. Nothing is posted automatically.

After the workshop, stop/delete the model through the dashboard and cancel unused runs. If you used programmatic deployment, also remove its runtime and the model-specific `*-storage` ServiceAccount, Role and RoleBinding when no longer needed. Remove only your own experiment's S3 prefix and artifacts; the sample does not delete shared storage or Connections.

## Offline checks and preparing notebooks for a PR

Run checks from the repository root after installing `automl/requirements.txt`. They validate data boundaries, pipeline arguments, state resumption, metric-based selection, artifact handling and the REST contract without cluster credentials.

To run only the behavioral checks:

```sh
python -m unittest discover -s automl/tests -k WorkflowTests -v
```

The full suite also validates notebook structure and Python syntax. It accepts saved example outputs:

```sh
python -m unittest discover -s automl/tests -v
```

Review the notebook diff and leave `START_NEW_RUN = False` before committing. Keep the original run evidence in the ignored `outputs/` directory.
