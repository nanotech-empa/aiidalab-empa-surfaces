class StructureBrowser(ipw.VBox):
    
    def __init__(self):
        # Find all process labels
        qb = QueryBuilder()
        qb.append(WorkCalculation,
                  project="attributes._process_label",
                  filters={
                      'attributes': {'!has_key': 'source_code'}
                  }
        )
        qb.order_by({WorkCalculation:{'ctime':'desc'}})
        process_labels = []
        for i in qb.iterall():
            if i[0] not in process_labels:
                process_labels.append(i[0])

        layout = ipw.Layout(width="900px")

        self.mode = ipw.RadioButtons(options=['all', 'uploaded', 'edited', 'calculated'], layout=ipw.Layout(width="100px"))
        
        
        self.age_range = ipw.IntRangeSlider(value=[0, 7], min=0, max=100, step=1, continuous_update=False,
                                            description='age in days:', layout=ipw.Layout(width="800px"))

        # Labels
        self.drop_label = ipw.Dropdown(options=(['All',] + process_labels),
                                       description='Process Label',
                                       style = {'description_width': '120px'})

        self.age_range.observe(self.search, names='value')
        self.mode.observe(self.search, names='value')
        self.drop_label.observe(self.search, names='value')
        
        box = ipw.HBox([self.mode, self.age_range])
        
        self.results = ipw.Dropdown(layout=layout)
        border = ipw.Layout(border="2px solid black")
        self.search()
        super(StructureBrowser, self).__init__([box, self.drop_label, self.results], layout=border)
    
    
    def preprocess(self):
        qb = QueryBuilder()
        qb.append(StructureData, filters={'extras': {'!has_key': 'formula'}})
        for n in qb.all(): # iterall() would interfere with set_extra()
            formula = n[0].get_formula()
            n[0].set_extra("formula", formula)

    
    def search(self, c=None):
        self.preprocess()
        
        qb = QueryBuilder()
        min_age = datetime.datetime.now() - datetime.timedelta(days=self.age_range.value[0])
        max_age = datetime.datetime.now() - datetime.timedelta(days=self.age_range.value[1])
        filters = {}
        filters["ctime"] = {'and':[{'<=': min_age},{'>': max_age}]}
        
        if self.drop_label.value != 'All':
            qb.append(WorkCalculation,
                      filters={
                          'attributes._process_label': self.drop_label.value
                      })
            
            qb.append(JobCalculation,
                      output_of=WorkCalculation)
            
            qb.append(StructureData,
                      output_of=JobCalculation,
                      filters=filters)
        else:        
            if self.mode.value == "uploaded":
                qb2 = QueryBuilder()
                qb2.append(StructureData, project=["id"])
                qb2.append(Node, input_of=StructureData)
                processed_nodes = [n[0] for n in qb2.all()]
                if processed_nodes:
                    filters['id'] = {"!in":processed_nodes}
                qb.append(StructureData, filters=filters)

            elif self.mode.value == "calculated":
                qb.append(JobCalculation)
                qb.append(StructureData, output_of=JobCalculation, filters=filters)

            elif self.mode.value == "edited":
                qb.append(WorkCalculation)
                qb.append(StructureData, output_of=WorkCalculation, filters=filters)

            else:
                self.mode.value == "all"
                qb.append(StructureData, filters=filters)

        qb.order_by({StructureData:{'ctime':'desc'}})
        matches = set([n[0] for n in qb.iterall()])
        matches = sorted(matches, reverse=True, key=lambda n: n.ctime)
        
        c = len(matches)
        options = OrderedDict()
        options["Select a Structure (%d found)"%c] = False

        for n in matches:
            label  = "PK: %d" % n.pk
            label += " | " + n.ctime.strftime("%Y-%m-%d %H:%M")
            label += " | " + n.get_extra("formula")
            label += " | " + n.description
            options[label] = n

        self.results.options = options