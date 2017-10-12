import json
import logging
import os

import requests

DEFAULT_SERVER = "http://ucca-demo.cs.huji.ac.il"
API_PREFIX = "/api/v1/"
SERVER_ADDRESS_ENV_VAR = "UCCA_APP_SERVER_ADDRESS"
AUTH_TOKEN_ENV_VAR = "UCCA_APP_AUTH_TOKEN"
EMAIL_ENV_VAR = "UCCA_APP_EMAIL"
PASSWORD_ENV_VAR = "UCCA_APP_PASSWORD"
PROJECT_ID_ENV_VAR = "UCCA_APP_PROJECT_ID"
SOURCE_ID_ENV_VAR = "UCCA_APP_SOURCE_ID"


class ServerAccessor(object):
    def __init__(self, server_address, email, password, auth_token, project_id, source_id, verbose, **kwargs):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        server_address = server_address or os.environ.get(SERVER_ADDRESS_ENV_VAR, DEFAULT_SERVER)
        self.prefix = server_address + API_PREFIX
        self.headers = {}  # Needed for self.request (login)
        token = auth_token or os.environ.get(AUTH_TOKEN_ENV_VAR) or self.login(
            email or os.environ[EMAIL_ENV_VAR], password or os.environ[PASSWORD_ENV_VAR])["token"]
        self.headers["Authorization"] = "Token " + token
        self.source = self.get_source(source_id or int(os.environ[SOURCE_ID_ENV_VAR]))
        self.project = self.get_project(project_id or int(os.environ[PROJECT_ID_ENV_VAR]))
        self.layer = self.get_layer(self.project["layer"]["id"])

    @staticmethod
    def add_arguments(argparser):
        argparser.add_argument("--server-address", help="UCCA-App server, otherwise set by " + SERVER_ADDRESS_ENV_VAR)
        argparser.add_argument("--email", help="UCCA-App email, otherwise set by " + EMAIL_ENV_VAR)
        argparser.add_argument("--password", help="UCCA-App password, otherwise set by " + PASSWORD_ENV_VAR)
        argparser.add_argument("--auth-token", help="authorization token (required only if email or password missing), "
                                                    "otherwise set by " + AUTH_TOKEN_ENV_VAR)
        argparser.add_argument("--project-id", type=int, help="project id, otherwise set by " + PROJECT_ID_ENV_VAR)
        argparser.add_argument("--source-id", type=int, help="source id, otherwise set by " + SOURCE_ID_ENV_VAR)
        argparser.add_argument("-v", "--verbose", action="store_true", help="detailed output")

    def request(self, method, url_suffix, **kwargs):
        response = requests.request(method, self.prefix + str(url_suffix), headers=self.headers, **kwargs)
        response.raise_for_status()
        return response

    def login(self, email, password):
        return self.request("post", "login", json=dict(email=email, password=password)).json()

    def get_source(self, source_id):
        logging.debug("Getting source %d" % source_id)
        source_out = self.request("get", "sources/%d/" % source_id).json()
        logging.debug("Got source: " + json.dumps(source_out))
        return source_out

    def get_project(self, project_id):
        logging.debug("Getting project %d" % project_id)
        project_out = self.request("get", "projects/%d/" % project_id).json()
        logging.debug("Got project: " + json.dumps(project_out))
        return project_out

    def get_layer(self, layer_id):
        logging.debug("Getting layer %d" % layer_id)
        layer_out = self.request("get", "layers/%d/" % layer_id).json()
        logging.debug("Got layer: " + json.dumps(layer_out))
        return layer_out

    def get_task(self, task_id):
        logging.debug("Getting task " + str(task_id))
        task_out = self.request("get", "tasks/" + str(task_id)).json()
        logging.debug("Got task: " + json.dumps(task_out))
        return task_out

    def get_user_task(self, task_id):
        logging.debug("Getting user task " + str(task_id))
        task_out = self.request("get", "user_tasks/" + str(task_id)).json()
        logging.debug("Got user task: " + json.dumps(task_out))
        return task_out

    def get_passage(self, passage_id):
        logging.debug("Getting passage " + str(passage_id))
        passage_out = self.request("get", "passages/" + str(passage_id)).json()
        logging.debug("Got passage: " + json.dumps(passage_out))
        return passage_out

    def create_passage(self, **kwargs):
        logging.debug("Creating passage: " + json.dumps(kwargs))
        passage_out = self.request("post", "passages/", json=kwargs).json()
        logging.debug("Created passage: " + json.dumps(passage_out))
        return passage_out

    def create_tokenization_task(self, **kwargs):
        logging.debug("Creating tokenization task: " + json.dumps(kwargs))
        tok_task_out = self.request("post", "tasks/", json=kwargs).json()
        logging.debug("Created tokenization task: " + json.dumps(tok_task_out))
        return tok_task_out

    def submit_tokenization_task(self, **kwargs):
        logging.debug("Submitting tokenization task: " + json.dumps(kwargs))
        self.request("put", "user_tasks/%d/draft" % kwargs["id"], json=kwargs)
        tok_user_task_out = self.request("put", "user_tasks/%d/submit" % kwargs["id"]).json()
        logging.debug("Submitted tokenization task: " + json.dumps(tok_user_task_out))
        return tok_user_task_out

    def create_annotation_task(self, **kwargs):
        logging.debug("Creating annotation task: " + json.dumps(kwargs))
        ann_task_out = self.request("post", "tasks/", json=kwargs).json()
        logging.debug("Created annotation task: " + json.dumps(ann_task_out))
        return ann_task_out

    def submit_annotation_task(self, **kwargs):
        logging.debug("Submitting annotation task: " + json.dumps(kwargs))
        self.request("put", "user_tasks/%d/draft" % kwargs["id"], json=kwargs)
        ann_user_task_out = self.request("put", "user_tasks/%d/submit" % kwargs["id"]).json()
        logging.debug("Submitted annotation task: " + json.dumps(ann_user_task_out))
        return ann_user_task_out
