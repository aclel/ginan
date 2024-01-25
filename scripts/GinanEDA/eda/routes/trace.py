import numpy as np
import json
from datetime import timedelta

from flask import Blueprint, current_app, render_template, request, session, make_response
import plotly.graph_objs as go


from backend.dbconnector.mongo import MongoDB
from backend.data.measurements import MeasurementArray, Measurements
from ..utilities import init_page, extra, generate_fig, generate_figs, aggregate_stats, get_data, get_arbitrary, get_keys_from_sub, get_distinct_vals
from . import eda_bp


@eda_bp.route("/advanced", methods=["GET", "POST"])
def trace():

    print(request.cookies)
    traceFormData   = json.loads(request.cookies["traceFormData"])
    mongoIdData     = json.loads(request.cookies["mongoIdData"])
    mongoValData    = json.loads(request.cookies["mongoValData"])
    mongoIdKeyVals  = json.loads(request.cookies["mongoIdKeyVals"])

    if request.method == "POST":
        return handle_post_request()
    # print("traceFormData")
    # print(traceFormData)
    # print("mongoIdData")
    # print(mongoIdData)
    # print("mongoValData")
    # print(mongoValData)

    return render_template(
        "trace.jinja",
        content=[],
        extra=extra,
        exlcude=0,
        selection=traceFormData,
        mongoIdData=mongoIdData,
        mongoValData=mongoValData,
        mongoIdKeyVals=mongoIdKeyVals)

    init_page(template="trace.jinja")


def handle_post_request():
    form_data = request.form
    form = {}
    form["ip"]      = form_data.get("ip")
    form["port"]    = int(form_data.get("port"))
    form["db"]      = form_data.get("db")
    form["type"]    = form_data.get("type")
    form["fCoeff"]  = form_data.get("fCoeff")
    form["filter"]  = form_data.get("filter")
    form["group"]   = form_data.getlist("group")
    form["datax"]   = form_data.getlist("datax")
    form["xaxis"]  = form_data.get("xaxis")
    form["yaxis"]  = form_data.get("yaxis")

    xaxis = form["xaxis"]
    yaxis = form["yaxis"]

    matchh = ""
    for key in form_data:
        if key[0:7] == "match--":
            val = form_data.get(key)
            form[key] = val
            if len(val):
                if len(matchh):
                    matchh += ","
                matchh += "\"" + key[7:] + "\":" + val

    group = ""
    for thing in form["group"]:
        if len(group):
            group += ","
        group += "\"" + thing + "\":1"

    # print(matchh)
    # print(group)

    traceFormData   = json.loads(request.cookies["traceFormData"])
    mongoIdData     = json.loads(request.cookies["mongoIdData"])
    mongoValData    = json.loads(request.cookies["mongoValData"])
    mongoIdKeyVals  = json.loads(request.cookies["mongoIdKeyVals"])


    traces = []

    if (form_data.get("update") == "Update"):

        current_app.logger.info("Updating")

        idKeys  = get_keys_from_sub(form["ip"], form["port"], form["db"], "Trace", "id");
        valKeys = get_keys_from_sub(form["ip"], form["port"], form["db"], "Trace", "val");

        idKeyVals = {}

        for idKey in idKeys:
            distincts = get_distinct_vals(form["ip"], form["port"], form["db"], "Trace", "id." + idKey)
            distincts.sort()
            idKeyVals[idKey] = distincts





        idKeys.sort()
        valKeys.sort()

        mongoIdData = json.dumps(idKeys)
        mongoValData = json.dumps(valKeys)
        mongoIdKeyVals = json.dumps(idKeyVals)

        renderedTemplate = render_template(
            "trace.jinja",
            # content=client.mongo_content,
            extra=extra,
            graphJSON=generate_figs(traces),
            mode="plotly2",
            selection=form,
            mongoIdData=idKeys,
            mongoValData=valKeys,
            mongoIdKeyVals=idKeyVals)

        response = make_response(renderedTemplate)
        response.set_cookie("mongoIdData",    mongoIdData,     max_age=timedelta(days=400))
        response.set_cookie("mongoValData",   mongoValData,    max_age=timedelta(days=400))
        response.set_cookie("mongoIdKeyVals", mongoIdKeyVals,  max_age=timedelta(days=400))

        return response

    if (form_data.get("plot") == "Plot"):

        current_app.logger.info("Plotting")

        data = MeasurementArray()
    # for series in form["series"] :
        # db_, series_ = series.split("\\")

        datas = []
        for i in range(len(form["datax"])):
            print("-------------------")
            print(form["datax"][i])
            datas.append( get_arbitrary(form["ip"], form["port"], form["db"], "Trace", matchh, group, form["datax"][i]))

        message = ""

        if len(datas) == 0:
            message = "Error getting data: No data"
        for datax in datas:
            if "excessFields" in datax:
                message="Excess data received, try grouping by one or more of: " + str(datax["excessFields"])

        if len(message):

            return render_template(
                "trace.jinja",
                selection=form,
                mongoIdData=mongoIdData,
                mongoValData=mongoValData,
                mongoIdKeyVals=mongoIdKeyVals,
                # content=client.mongo_content,
                extra=extra,
                message=message)

        mode = "markers" if form["type"] == "Scatter" else "lines"
        if form["type"] == "QQ":
            data.compute_qq()
            mode = "markers"
        table = {}
        current_app.logger.warning("starting plots")

        for datax in datas:
          tracesX = []
          for label, traceData in datax.items():
            # print("trace:" + label)
            traceX = []
            traceY = []

            # print("data: " + str(traceData))

            for element in traceData:

                element["y"] = element["y"][0]
                # print(element)

                if xaxis[0] == '_':
                    x = element["_id"][xaxis[1:]]
                else:
                    x = element       [xaxis]

                if yaxis[0] == '_':
                    y = element["_id"][yaxis[1:]]
                else:
                    y = element       [yaxis]

                try:
                    lpf
                except:
                    lpf         = y
                    last        = y
                    lastLast    = y

                # print(element[yaxis])
                # print(str(x)  + " " + str(val))
                if type(y) == int or type(y) == float:

                    lpf = lpf + (y - lpf) * float(form["fCoeff"])
                    hpf = y - lpf
                    if (form["filter"] == "HPF"):
                        y = hpf
                    if (form["filter"] == "LPF"):
                        y = lpf
                    if (form["filter"] == "DIFF"):
                        y = y - last
                    if (form["filter"] == "DIFF2"):
                        y = y - 2 * last + lastLast

                lastLast = last
                last = y

                traceX.append(x)
                traceY.append(y)

            x_hover_template = "%{x}<br>"
            tracesX.append(
                go.Scatter(
                    x=traceX,
                    y=traceY,
                    mode=mode,
                    name=f"{label}",
                    hovertemplate=x_hover_template + "%{y:.4e%}<br>" + str(element["y"]) + "<br>" + f"{label}",
                    legendgroup='group1'
                )
            )
          traces.append(tracesX)

        current_app.logger.warning("end plots")

        renderedTemplate = render_template(
            "trace.jinja",
            # content=client.mongo_content,
            extra=extra,
            graphJSON=generate_figs(traces),
            mode="plotly2",
            selection=form,
            mongoIdData=mongoIdData,
            mongoValData=mongoValData,
            mongoIdKeyVals=mongoIdKeyVals)

        response = make_response(renderedTemplate)
        response.set_cookie("traceFormData", json.dumps(form), max_age=timedelta(days=400))

        return response

