import re
import sys
from collections import Counter
from xml.etree.ElementTree import fromstring

import psycopg2

from ucca import convert, layer1
from ucca.layer1 import EdgeTags as ET
from ucca.layer1 import NodeTags as NT

UNK_LINKAGE_TYPE = 'UNK'
PLACE_HOLDER = "%s"


##############################################################################
# Returns the most recent xmls from db with a passage id pid and usernames
# (a list). The xmls are ordered in the same way as the list usernames.
##############################################################################


def get_connection(db, host):
    """ connects to the db and host, returns a connection object """
    return psycopg2.connect(host=host, database=db)


def get_cursor_con(con):
    """ gets a connection and create a cursur to the search path"""
    cur = con.cursor()
    cur.execute("SET search_path TO oabend")
    return cur


def get_cursor(db, host):
    """ create a cursur to the search path
        NOTE: if you have a connection up use get_cursor_con instead
    """
    return get_cursor_con(get_connection(db, host))


def get_xml_trees(db, host, pid, usernames, graceful=False):
    """ Params: db, host, paragraph id, the list of usernames wanted,
        Optional:
        graceful: True if no excpetions are to be raised
        excpetion raised if a user did not submit an annotation for the passage
        returns a list of xml roots elements
        """
    cur = get_cursor(db, host)
    xmls = []
    for username in usernames:
        username = str(username)  # precaution for cases bad input e.g. 101
        cur_uid = get_uid(db, host, username)
        cur.execute("SELECT xml FROM xmls WHERE paid=" + PLACE_HOLDER +
                    " AND uid=" + PLACE_HOLDER + "ORDER BY ts DESC",
                    (pid, cur_uid))
        raw_xml = cur.fetchone()
        if not raw_xml and not graceful:
            raise Exception("The user " + username +
                            " did not submit an annotation for this passage")
        else:
            xmls.append(fromstring(raw_xml[0]))
    return xmls


def get_by_xid(db, host, xid, graceful=False):
    """Returns the passages that correspond to the xid

        Optional:
        graceful: True if no excpetions are to be raised
        excpetion raised if xid does no exist
    """
    # precaution for bad input e.g. 101->'101'
    xid = str(xid)
    cur = get_cursor(db, host)
    cur.execute("SELECT xml FROM xmls WHERE id" + "=" +
                PLACE_HOLDER, (int(xid),))
    raw_xml = cur.fetchone()
    if not raw_xml and not graceful:
        raise Exception("The xid " + xid + " does not exist")
    elif raw_xml:
        return fromstring(raw_xml[0])


def get_by_xids(db, host, xids, graceful=False):
    """Returns the passages that correspond to iterable xids

        Optional:
        graceful: True if no excpetions are to be raised
        excpetion raised if xid does no exist
    """
    _ = get_cursor(db, host)
    xmls = [get_by_xid(db, host, xid, graceful) for xid in xids]
    return xmls


def get_uid(db, host, username, operator="=", graceful=False):
    """Returns the uid matching the given username.
        Optional:
        operator: default operator is '='
        graceful: True if no excpetions are to be raised
        excpetion raised if a user does no exist
        """
    cur = get_cursor(db, host)
    username = str(username)
    cur.execute("SELECT id FROM users WHERE username" + operator +
                PLACE_HOLDER, (username,))
    cur_uid = cur.fetchone()
    if not cur_uid and not graceful:
        raise Exception("The user " + username + " does not exist")
    return cur_uid[0]


def write_to_db(db, host, xml, new_pid, new_prid, username,
                operator="=", graceful=False):
    """writes to db
        Optional:
        operator: default operator is '='
        graceful: True if no excpetions are to be raised
        excpetion raised if a user does no exist
    """
    print("warning this function was not tested and better" +
          " API specification (parameters what is written) is needed")
    con = get_connection()
    cur_uid = get_uid(db, host, username, operator, graceful)
    now = datetime.datetime.now()
    con.execute("INSERT INTO xmls VALUES (NULL, " +
                (PLACE_HOLDER + ', ') * 5 + "0, " +
                PLACE_HOLDER + ")", (xml, new_pid, prid, cur_uid, '', now))


def print_most_recent_xids(db, host, username, n=10):
    """print the most recent xids of the given username."""
    cur_uid = get_uid(db, host, username)
    cur = get_cursor(db, host)
    cur.execute("SELECT id, paid FROM xmls WHERE uid=" + PLACE_HOLDER +
                " ORDER BY ts DESC", (cur_uid,))
    print(username)
    print("=============")
    for xid in get_most_recent_xids(db, host, username, n):
        print(xid)


def get_most_recent_xids(db, host, username, n=10):
    """Returns the n most recent xids of the given username."""
    cur_uid = get_uid(db, host, username)
    cur = get_cursor(db, host)
    cur.execute("SELECT id, paid FROM xmls WHERE uid=" + PLACE_HOLDER +
                " ORDER BY ts DESC", (cur_uid,))
    r = [cur.fetchone() for _ in range(n)]
    return r


def get_passage(db, host, pid):
    """Returns the passages with the given id numbers"""
    cur = get_cursor(db, host)
    cur.execute("SELECT passage FROM passages WHERE id=" +
                PLACE_HOLDER, (pid,))
    output = cur.fetchone()
    if not output:
        raise Exception("No passage with ID=" + pid)
    return output[0]


def linkage_type(u):
    """
    Returns the type of the primary linkage
    the scene participates in.
    It can be A,E or H.
    If it is a C, it returns the tag of
    the first fparent which is an A,E or H.
    If it does not find an fparent
    with either of these categories,
    it returns UNK_LINKAGE_TYPE.
    """
    cur_u = u
    while cur_u:
        if cur_u.ftag in [ET.Participant, ET.Elaborator, ET.ParallelScene]:
            return cur_u.ftag
        elif cur_u.ftag != ET.Center:
            return UNK_LINKAGE_TYPE
        else:
            cur_u = cur_u.fparent
    return UNK_LINKAGE_TYPE


def unit_length(u):
    """
    Returns the number of terminals
    (excluding remote units and punctuations)
    that are descendants of the unit u.
    """
    return len(u.get_terminals(punct=False, remotes=False))


def get_tasks(db, host, username):
    """
    Returns for that user a list of submitted passages
    and a list of assigned but not submitted passages.
    Each passage is given in the format:
    (<passage ID>, <source>, <recent submitted xid or -1 if not submitted>,
    <number of tokens in the passage>,
     <number of units in the passage>, <number of scenes in the passage>,
    <average length of a scene>).
     It also returns a distribution of the categories.
    """
    output_submitted = []
    category_distribution = Counter()
    # the categories of scenes. can be A, E or H
    scene_distribution = Counter()

    uid = get_uid(db, username)
    cur = get_cursor(db, username)
    cur.execute("SELECT pid,status FROM tasks WHERE uid=" +
                PLACE_HOLDER, (uid,))
    r = cur.fetchall()
    submitted_paids = [x[0] for x in r if x[1] == 1]
    incomplete_paids = [x[0] for x in r if x[1] == 0]

    wspace = re.compile("\\s+")

    for paid in submitted_paids:
        sum_scene_length = 0
        if paid < 100:  # skipping training passages
            continue
        cur.execute("SELECT passage,source FROM passages WHERE id=" +
                    PLACE_HOLDER, (paid,))
        r = cur.fetchone()
        if r:
            num_tokens = len(wspace.split(r[0])) - 1
            source = r[1]
            cur.execute("SELECT id, xml FROM xmls WHERE paid=" +
                        PLACE_HOLDER + " AND uid=" + PLACE_HOLDER +
                        " AND status=" + PLACE_HOLDER +
                        " ORDER BY ts DESC", (paid, uid, 1))
            r = cur.fetchone()
            if r:
                xid = r[0]
                # noinspection PyBroadException
                try:
                    ucca_dag = convert.from_site(fromstring(r[1]))
                except Exception:
                    sys.stderr.write("Skipped.\n")
                    continue
                num_units = len([x for x in ucca_dag.layer(layer1.LAYER_ID).all
                                 if x.tag == NT.Foundational]) - 1
                for node in ucca_dag.layer(layer1.LAYER_ID).all:
                    category_distribution.update([e.tag for e in node
                                                  if e.tag
                                                  not in
                                                  [ET.Punctuation, ET.LinkArgument, ET.LinkRelation, ET.Terminal]])
                # getting the scene categories
                scenes = [x for x in ucca_dag.layer(layer1.LAYER_ID).all
                          if x.tag == NT.Foundational and x.is_scene()]
                scene_distribution.update([linkage_type(sc) for sc in scenes])
                sum_scene_length += sum([unit_length(x) for x in scenes])

        output_submitted.append((paid, source, xid,
                                 num_tokens, num_units, len(scenes),
                                 1.0 * sum_scene_length / len(scenes)))

    return output_submitted, category_distribution, scene_distribution
