"""Create a heterogeneous version of the merged CHESS database.

For each table used by experimental queries, this script:
1. Creates a COPY of the table with renamed columns and table name
2. Moves 50% of the rows from the original to the renamed copy
3. Generates database_description CSVs for the new tables

Purpose: Test whether CHESS (Text2SQL) can handle schema heterogeneity -
the same data split across differently-named tables. VKG/ontology-based
approaches handle this via semantic mappings; CHESS must discover the
relationship from table/column names alone.

Usage:
    python scripts/create_chess_heterogeneous_db.py
"""

import csv
import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHESS_DB_DIR = PROJECT_ROOT / "tools" / "chess" / "data" / "dev" / "dev_databases"
SOURCE_DB = CHESS_DB_DIR / "merged" / "merged.sqlite"
TARGET_DIR = CHESS_DB_DIR / "merged_heterogeneous"
TARGET_DB = TARGET_DIR / "merged_heterogeneous.sqlite"
TARGET_DESC_DIR = TARGET_DIR / "database_description"

# Source description dir (to copy unchanged tables)
SOURCE_DESC_DIR = CHESS_DB_DIR / "merged" / "database_description"

# =============================================================================
# RENAME MAPPINGS: original_table -> {new_name, columns: {old_col -> new_col}}
# All 109 tables have hand-crafted, semantically meaningful renames.
# =============================================================================

RENAME_MAP = {
    # =========================================================================
    # EDU Dataset
    # =========================================================================
    "undergraduateStudent": {
        "new_name": "bachelor_participants",
        "columns": {
            "nr": "participant_id",
            "name": "full_name",
            "telephone": "phone_number",
            "emailAddress": "email",
            "age": "student_age",
            "advisor": "supervisor_id",
            "memberOf": "dept_id",
        },
        "description": "Bachelor-level student participants in academic programs",
    },
    "graduateStudent": {
        "new_name": "postgrad_scholars",
        "columns": {
            "nr": "scholar_id",
            "name": "full_name",
            "telephone": "phone_number",
            "emailAddress": "email",
            "age": "scholar_age",
            "undergraduateDegreeFrom": "bachelors_from",
            "advisor": "supervisor_id",
            "memberOf": "dept_id",
        },
        "description": "Postgraduate research scholars and their affiliations",
    },
    "undergraduateCourse": {
        "new_name": "bachelor_modules",
        "columns": {
            "nr": "module_id",
            "name": "module_title",
            "teacher": "lecturer_id",
            "teachingAssistant": "tutor_id",
        },
        "description": "Undergraduate teaching modules and their instructors",
    },
    "graduateCourse": {
        "new_name": "postgrad_seminars",
        "columns": {
            "nr": "seminar_id",
            "name": "seminar_title",
            "teacher": "lecturer_id",
        },
        "description": "Graduate-level seminar courses",
    },
    "undergraduateStudentTakeCourse": {
        "new_name": "bachelor_enrollments",
        "columns": {
            "undergraduateStudentID": "participant_id",
            "undergraduateCourseID": "module_id",
        },
        "description": "Enrollment records linking bachelor participants to modules",
    },
    "graduateStudentTakeCourse": {
        "new_name": "postgrad_registrations",
        "columns": {
            "graduateStudentID": "scholar_id",
            "graduateCourseID": "seminar_id",
        },
        "description": "Registration records for postgraduate seminar attendance",
    },
    "faculty": {
        "new_name": "academic_staff",
        "columns": {
            "nr": "staff_id",
            "name": "full_name",
            "telephone": "phone",
            "emailAddress": "email",
            "undergraduateDegreeFrom": "ug_degree_from",
            "masterDegreeFrom": "masters_from",
            "doctoralDegreeFrom": "phd_from",
            "worksFor": "dept_id",
        },
        "description": "Academic staff members and their departmental assignments",
    },
    "professor": {
        "new_name": "senior_academics",
        "columns": {
            "nr": "academic_id",
            "professorType": "academic_rank",
            "researchInterest": "research_area",
            "headOf": "leads_dept",
        },
        "description": "Senior academic personnel with research leadership roles",
    },
    "department": {
        "new_name": "organizational_units",
        "columns": {
            "nr": "unit_id",
            "name": "unit_name",
            "subOrganizationOf": "parent_org_id",
        },
        "description": "Organizational units within academic institutions",
    },
    "university": {
        "new_name": "academic_institutions",
        "columns": {
            "nr": "institution_id",
            "name": "institution_name",
        },
        "description": "Higher education institutions",
    },
    "publication": {
        "new_name": "research_outputs",
        "columns": {
            "nr": "output_id",
            "name": "output_name",
            "title": "output_title",
            "abstract": "summary",
            "mainAuthor": "primary_author_id",
        },
        "description": "Published research outputs and their authors",
    },
    "coAuthorOfPublication": {
        "new_name": "collaborative_authorships",
        "columns": {
            "publicationID": "output_id",
            "graduateStudentID": "scholar_id",
        },
        "description": "Co-authorship links between scholars and research outputs",
    },
    "researchGroup": {
        "new_name": "research_teams",
        "columns": {
            "nr": "team_id",
            "subOrganizationOf": "parent_dept_id",
        },
        "description": "Research teams organized under departments",
    },
    "lecturer": {
        "new_name": "teaching_staff",
        "columns": {
            "nr": "instructor_id",
        },
        "description": "Teaching staff with instructional roles",
    },

    # =========================================================================
    # TRN Dataset (GTFS)
    # =========================================================================
    "ROUTES": {
        "new_name": "transit_lines",
        "columns": {
            "route_id": "line_id",
            "agency_id": "operator_id",
            "route_short_name": "line_code",
            "route_long_name": "line_name",
            "route_desc": "description",
            "route_type": "transport_mode",
            "route_url": "info_url",
            "route_color": "color_hex",
            "route_text_color": "text_color",
        },
        "description": "Public transit lines operated by transport agencies",
    },
    "TRIPS": {
        "new_name": "transit_journeys",
        "columns": {
            "trip_id": "journey_id",
            "route_id": "line_id",
            "service_id": "schedule_id",
            "trip_headsign": "destination",
            "trip_short_name": "journey_code",
            "direction_id": "direction",
            "block_id": "block",
            "shape_id": "path_id",
            "wheelchair_accessible": "accessible",
        },
        "description": "Individual transit journeys along defined lines",
    },
    "STOP_TIMES": {
        "new_name": "schedule_events",
        "columns": {
            "trip_id": "journey_id",
            "arrival_time": "arrive_at",
            "departure_time": "depart_at",
            "stop_id": "station_id",
            "stop_sequence": "sequence_no",
            "stop_headsign": "local_destination",
            "pickup_type": "boarding_type",
            "drop_off_type": "alighting_type",
            "shape_dist_traveled": "distance",
        },
        "description": "Scheduled arrival and departure events at transit stations",
    },
    "STOPS": {
        "new_name": "transit_stations",
        "columns": {
            "stop_id": "station_id",
            "stop_code": "station_code",
            "stop_name": "station_name",
            "stop_desc": "description",
            "stop_lat": "latitude",
            "stop_lon": "longitude",
            "zone_id": "fare_zone",
            "stop_url": "info_url",
            "location_type": "type",
            "parent_station": "parent_id",
            "stop_timezone": "timezone",
            "wheelchair_boarding": "accessible",
        },
        "description": "Physical transit station locations and metadata",
    },

    # =========================================================================
    # NRG Dataset (Norwegian Petroleum)
    # =========================================================================
    "field": {
        "new_name": "petroleum_deposits",
        "columns": {
            "fldName": "deposit_name",
            "cmpLongName": "operator_name",
            "fldCurrentActivitySatus": "activity_status",
            "wlbName": "discovery_well",
            "wlbCompletionDate": "well_completion_date",
            "fldOwnerKind": "owner_type",
            "fldOwnerName": "owner_name",
            "fldNpdidOwner": "owner_id",
            "fldNpdidField": "deposit_id",
            "wlbNpdidWellbore": "wellbore_id",
            "cmpNpdidCompany": "company_id",
            "fldFactPageUrl": "fact_page_url",
            "fldFactMapUrl": "fact_map_url",
            "fldDateUpdated": "date_updated",
            "fldDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Petroleum deposit fields on the Norwegian continental shelf",
    },
    "company": {
        "new_name": "petroleum_companies",
        "columns": {
            "cmpLongName": "company_name",
            "cmpOrgNumberBrReg": "org_number",
            "cmpGroup": "company_group",
            "cmpShortName": "short_name",
            "cmpNpdidCompany": "company_id",
            "cmpLicenceOperCurrent": "current_operator_licences",
            "cmpLicenceOperFormer": "former_operator_licences",
            "cmpLicenceLicenseeCurrent": "current_licensee_licences",
            "cmpLicenceLicenseeFormer": "former_licensee_licences",
            "dateSyncNPD": "sync_date",
        },
        "description": "Oil and gas companies operating on Norwegian shelf",
    },
    "field_production_yearly": {
        "new_name": "annual_output",
        "columns": {
            "prfInformationCarrier": "field_name",
            "prfYear": "production_year",
            "prfPrdOilNetMillSm3": "oil_output_msm3",
            "prfPrdGasNetBillSm3": "gas_output_bsm3",
            "prfPrdNGLNetMillSm3": "ngl_output_msm3",
            "prfPrdCondensateNetMillSm3": "condensate_msm3",
            "prfPrdOeNetMillSm3": "total_oe_msm3",
            "prfPrdProducedWaterInFieldMillSm3": "produced_water_msm3",
            "prfNpdidInformationCarrier": "field_id",
        },
        "description": "Annual petroleum production output by field",
    },
    "field_operator_hst": {
        "new_name": "deposit_operator_history",
        "columns": {
            "fldName": "deposit_name",
            "cmpLongName": "operator_name",
            "fldOperatorFrom": "operator_from_date",
            "fldOperatorTo": "operator_to_date",
            "fldNpdidField": "deposit_id",
            "cmpNpdidCompany": "company_id",
            "fldOperatorDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical operator assignments for petroleum deposits",
    },
    "field_reserves": {
        "new_name": "deposit_reserves",
        "columns": {
            "fldName": "deposit_name",
            "fldRecoverableOil": "recoverable_oil",
            "fldRecoverableGas": "recoverable_gas",
            "fldRecoverableNGL": "recoverable_ngl",
            "fldRecoverableCondensate": "recoverable_condensate",
            "fldRecoverableOE": "recoverable_oe",
            "fldRemainingOil": "remaining_oil",
            "fldRemainingGas": "remaining_gas",
            "fldRemainingNGL": "remaining_ngl",
            "fldRemainingCondensate": "remaining_condensate",
            "fldRemainingOE": "remaining_oe",
            "fldDateOffResEstDisplay": "estimate_date",
            "fldNpdidField": "deposit_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Estimated petroleum reserves by deposit field",
    },
    "wellbore_exploration_all": {
        "new_name": "exploration_boreholes",
        "columns": {
            "wlbWellboreName": "borehole_name",
            "wlbWell": "well_name",
            "wlbDrillingOperator": "drilling_company",
            "wlbDrillingOperatorGroup": "drilling_group",
            "wlbProductionLicence": "licence",
            "wlbPurpose": "purpose",
            "wlbStatus": "status",
            "wlbContent": "content",
            "wlbWellType": "well_type",
            "wlbEntryDate": "entry_date",
            "wlbCompletionDate": "completion_date",
            "wlbField": "field_name",
            "wlbTotalDepth": "total_depth",
            "wlbFinalVerticalDepth": "vertical_depth",
            "wlbWaterDepth": "water_depth",
            "wlbMainArea": "main_area",
            "wlbNpdidWellbore": "borehole_id",
            "wlbEntryYear": "entry_year",
            "wlbCompletionYear": "completion_year",
        },
        "description": "Exploration boreholes drilled on the Norwegian continental shelf",
        "partial_columns": True,  # Only rename subset, keep rest as-is
    },
    "wellbore_development_all": {
        "new_name": "development_drills",
        "columns": {
            "wlbWellboreName": "drill_name",
            "wlbWell": "well_name",
            "wlbDrillingOperator": "drilling_company",
            "wlbProductionLicence": "licence",
            "wlbWellType": "well_type",
            "wlbEntryDate": "entry_date",
            "wlbCompletionDate": "completion_date",
            "wlbField": "field_name",
            "wlbTotalDepth": "total_depth",
            "wlbFinalVerticalDepth": "vertical_depth",
            "wlbWaterDepth": "water_depth",
            "wlbMainArea": "main_area",
            "wlbNpdidWellbore": "drill_id",
            "wlbEntryYear": "entry_year",
            "wlbCompletionYear": "completion_year",
        },
        "description": "Development drilling wells for petroleum production",
        "partial_columns": True,
    },

    # =========================================================================
    # BSBM Dataset (Berlin SPARQL Benchmark)
    # =========================================================================
    "product": {
        "new_name": "merchandise",
        "columns": {
            "nr": "item_id",
            "label": "item_name",
            "comment": "description",
            "producer": "manufacturer_id",
            "propertyNum1": "spec_num1",
            "propertyNum2": "spec_num2",
            "propertyNum3": "spec_num3",
            "propertyNum4": "spec_num4",
            "propertyNum5": "spec_num5",
            "propertyNum6": "spec_num6",
            "propertyTex1": "spec_text1",
            "propertyTex2": "spec_text2",
            "propertyTex3": "spec_text3",
            "propertyTex4": "spec_text4",
            "propertyTex5": "spec_text5",
            "propertyTex6": "spec_text6",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Merchandise items with specifications and manufacturer details",
    },
    "offer": {
        "new_name": "price_listings",
        "columns": {
            "nr": "listing_id",
            "product": "item_id",
            "producer": "manufacturer_id",
            "vendor": "seller_id",
            "price": "unit_price",
            "validFrom": "effective_from",
            "validTo": "effective_to",
            "deliveryDays": "shipping_days",
            "offerWebpage": "listing_url",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Price listings for merchandise items from various sellers",
    },
    "review": {
        "new_name": "customer_reviews",
        "columns": {
            "nr": "review_id",
            "product": "item_id",
            "producer": "manufacturer_id",
            "person": "reviewer_id",
            "reviewDate": "review_date",
            "title": "headline",
            "text": "body",
            "language": "lang",
            "rating1": "quality_score",
            "rating2": "value_score",
            "rating3": "delivery_score",
            "rating4": "overall_score",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Customer reviews and ratings for merchandise items",
    },
    "person": {
        "new_name": "customers",
        "columns": {
            "nr": "customer_id",
            "name": "customer_name",
            "mbox_sha1sum": "email_hash",
            "country": "country_code",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Customer profiles who write reviews",
    },
    "productfeature": {
        "new_name": "item_specs",
        "columns": {
            "nr": "spec_id",
            "label": "spec_name",
            "comment": "description",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Specification attributes for merchandise categorization",
    },
    "productfeatureproduct": {
        "new_name": "item_spec_links",
        "columns": {
            "product": "item_id",
            "productFeature": "spec_id",
        },
        "description": "Links between merchandise items and their specifications",
    },
    "producttype": {
        "new_name": "item_categories",
        "columns": {
            "nr": "category_id",
            "label": "category_name",
            "comment": "description",
            "parent": "parent_category_id",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Hierarchical categories for merchandise classification",
    },
    "producttypeproduct": {
        "new_name": "item_category_links",
        "columns": {
            "product": "item_id",
            "productType": "category_id",
        },
        "description": "Links between merchandise items and their categories",
    },
    "producer": {
        "new_name": "manufacturers",
        "columns": {
            "nr": "mfr_id",
            "label": "mfr_name",
            "comment": "mfr_description",
            "homepage": "website",
            "country": "country_code",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Manufacturers who produce merchandise items",
    },
    "vendor": {
        "new_name": "sellers",
        "columns": {
            "nr": "seller_id",
            "label": "seller_name",
            "comment": "seller_description",
            "homepage": "website",
            "country": "country_code",
            "publisher": "pub_id",
            "publishDate": "pub_date",
        },
        "description": "Sellers who offer merchandise for purchase",
    },

    # =========================================================================
    # LCA Dataset (Life Cycle Assessment)
    # =========================================================================
    "hdpe_flows": {
        "new_name": "lifecycle_exchanges",
        "columns": {
            "id": "exchange_id",
            "activity": "process_name",
            "flow": "flow_name",
            "flow_type": "exchange_type",
            "flow_direction": "direction",
            "kategorie": "category",
            "measure": "quantity",
            "unit": "unit_of_measure",
        },
        "description": "Material and energy exchanges in lifecycle processes",
    },
    "characterisation_factors": {
        "new_name": "impact_factors",
        "columns": {
            "id": "factor_id",
            "elementary_flow": "flow_name",
            "ef_categories": "categories",
            "cf": "factor_value",
            "cf_unit": "factor_unit",
        },
        "description": "Environmental impact characterisation factors for flows",
    },

    # =========================================================================
    # TRN Dataset - remaining GTFS tables
    # =========================================================================
    "AGENCY": {
        "new_name": "transport_operators",
        "columns": {
            "agency_id": "operator_id",
            "agency_name": "operator_name",
            "agency_url": "website",
            "agency_timezone": "timezone",
            "agency_lang": "language",
            "agency_phone": "phone",
            "agency_fare_url": "fare_info_url",
        },
        "description": "Public transport operating companies",
    },
    "CALENDAR": {
        "new_name": "service_schedules",
        "columns": {
            "service_id": "schedule_id",
            "monday": "runs_monday",
            "tuesday": "runs_tuesday",
            "wednesday": "runs_wednesday",
            "thursday": "runs_thursday",
            "friday": "runs_friday",
            "saturday": "runs_saturday",
            "sunday": "runs_sunday",
            "start_date": "valid_from",
            "end_date": "valid_until",
        },
        "description": "Weekly service schedule patterns for transit operations",
    },
    "CALENDAR_DATES": {
        "new_name": "schedule_exceptions",
        "columns": {
            "service_id": "schedule_id",
            "date": "exception_date",
            "exception_type": "exception_kind",
        },
        "description": "Service schedule exceptions for holidays and special dates",
    },
    "FEED_INFO": {
        "new_name": "data_feed_metadata",
        "columns": {
            "id": "metadata_id",
            "feed_publisher_name": "publisher",
            "feed_publisher_url": "publisher_url",
            "feed_lang": "language",
            "feed_start_date": "coverage_start",
            "feed_end_date": "coverage_end",
            "feed_version": "version",
        },
        "description": "Metadata about the transit data feed and its publisher",
    },
    "FREQUENCIES": {
        "new_name": "headway_intervals",
        "columns": {
            "trip_id": "journey_id",
            "start_time": "interval_start",
            "end_time": "interval_end",
            "headway_secs": "frequency_seconds",
            "exact_times": "precise_timing",
        },
        "description": "Service frequency intervals for recurring transit journeys",
    },
    "SHAPES": {
        "new_name": "route_geometries",
        "columns": {
            "shape_id": "geometry_id",
            "shape_pt_lat": "latitude",
            "shape_pt_lon": "longitude",
            "shape_pt_sequence": "point_order",
            "shape_dist_traveled": "cumulative_distance",
        },
        "description": "Geographic path geometries for transit route visualization",
    },

    # =========================================================================
    # NRG Dataset - APA (Awards in Predefined Areas)
    # =========================================================================
    "apaAreaGross": {
        "new_name": "predefined_area_boundaries",
        "columns": {
            "apaMap_no": "map_number",
            "apaAreaGeometry_KML_WGS84": "geometry_kml",
            "apaAreaGross_id": "boundary_id",
        },
        "description": "Gross boundary geometries for predefined petroleum award areas",
    },
    "apaAreaNet": {
        "new_name": "predefined_area_blocks",
        "columns": {
            "blkId": "block_id",
            "blkLabel": "block_label",
            "qdrName": "quadrant_name",
            "blkName": "block_name",
            "prvName": "province_name",
            "apaAreaType": "area_type",
            "urlNPD": "registry_url",
            "apaAreaNet_id": "net_area_id",
        },
        "description": "Net area blocks within predefined petroleum award zones",
    },

    # =========================================================================
    # NRG Dataset - BAA (Business Arrangement Areas)
    # =========================================================================
    "baaArea": {
        "new_name": "business_arrangement_zones",
        "columns": {
            "baaNpdidBsnsArrArea": "arrangement_id",
            "baaNpdidBsnsArrAreaPoly": "polygon_id",
            "baaName": "arrangement_name",
            "baaKind": "arrangement_type",
            "baaAreaPolyDateValidFrom": "polygon_valid_from",
            "baaAreaPolyDateValidTo": "polygon_valid_to",
            "baaAreaPolyActive": "polygon_active",
            "baaDateApproved": "approval_date",
            "baaDateValidFrom": "valid_from",
            "baaDateValidTo": "valid_to",
            "baaActive": "is_active",
            "baaFactPageUrl": "fact_page_url",
            "baaFactMapUrl": "fact_map_url",
        },
        "description": "Business arrangement zone polygons on the continental shelf",
    },
    "bsns_arr_area": {
        "new_name": "commercial_agreements",
        "columns": {
            "baaName": "agreement_name",
            "baaKind": "agreement_type",
            "baaDateApproved": "approval_date",
            "baaDateValidFrom": "valid_from",
            "baaDateValidTo": "valid_to",
            "baaFactPageUrl": "fact_page_url",
            "baaFactMapUrl": "fact_map_url",
            "baaNpdidBsnsArrArea": "agreement_id",
            "baaDateUpdated": "date_updated",
            "baaDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Commercial arrangement agreements for petroleum operations",
    },
    "bsns_arr_area_area_poly_hst": {
        "new_name": "commercial_agreement_polygons",
        "columns": {
            "baaName": "agreement_name",
            "baaAreaPolyDateValidFrom": "polygon_valid_from",
            "baaAreaPolyDateValidTo": "polygon_valid_to",
            "baaAreaPolyNationCode2": "country_code",
            "baaAreaPolyBlockName": "block_name",
            "baaAreaPolyNo": "polygon_number",
            "baaAreaPolyArea": "polygon_area_km2",
            "baaNpdidBsnsArrArea": "agreement_id",
            "baaAreaPolyDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical polygon boundaries of commercial agreement areas",
    },
    "bsns_arr_area_licensee_hst": {
        "new_name": "commercial_agreement_partners",
        "columns": {
            "baaName": "agreement_name",
            "baaLicenseeDateValidFrom": "partnership_from",
            "baaLicenseeDateValidTo": "partnership_to",
            "cmpLongName": "company_name",
            "baaLicenseeInterest": "ownership_share",
            "baaLicenseeSdfi": "state_share",
            "baaNpdidBsnsArrArea": "agreement_id",
            "cmpNpdidCompany": "company_id",
            "baaLicenseeDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical partnership stakes in commercial arrangement areas",
    },
    "bsns_arr_area_operator": {
        "new_name": "commercial_agreement_managers",
        "columns": {
            "baaName": "agreement_name",
            "cmpLongName": "manager_company",
            "baaNpdidBsnsArrArea": "agreement_id",
            "cmpNpdidCompany": "company_id",
            "baaOperatorDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Operating companies managing commercial arrangement areas",
    },
    "bsns_arr_area_transfer_hst": {
        "new_name": "commercial_agreement_transfers",
        "columns": {
            "baaName": "agreement_name",
            "baaTransferDateValidFrom": "transfer_date",
            "baaTransferDirection": "transfer_direction",
            "baaTransferKind": "transfer_type",
            "cmpLongName": "company_name",
            "baaTransferredInterest": "transferred_share",
            "baaTransferSdfi": "state_share",
            "baaNpdidBsnsArrArea": "agreement_id",
            "cmpNpdidCompany": "company_id",
            "baaTransferDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Ownership transfer history for commercial arrangement areas",
    },

    # =========================================================================
    # NRG Dataset - Company & Discovery
    # =========================================================================
    "company_reserves": {
        "new_name": "corporate_resource_estimates",
        "columns": {
            "cmpLongName": "company_name",
            "fldName": "field_name",
            "cmpRecoverableOil": "recoverable_oil",
            "cmpRecoverableGas": "recoverable_gas",
            "cmpRecoverableNGL": "recoverable_ngl",
            "cmpRecoverableCondensate": "recoverable_condensate",
            "cmpRecoverableOE": "recoverable_oe_total",
            "cmpRemainingOil": "remaining_oil",
            "cmpRemainingGas": "remaining_gas",
            "cmpRemainingNGL": "remaining_ngl",
            "cmpRemainingCondensate": "remaining_condensate",
            "cmpRemainingOE": "remaining_oe_total",
            "cmpDateOffResEstDisplay": "estimate_display_date",
            "cmpShare": "company_share_pct",
            "fldNpdidField": "field_id",
            "cmpNpdidCompany": "company_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Resource reserve estimates allocated to companies per field",
    },
    "discovery": {
        "new_name": "hydrocarbon_finds",
        "columns": {
            "dscName": "find_name",
            "cmpLongName": "reporting_company",
            "dscCurrentActivityStatus": "activity_status",
            "dscHcType": "hydrocarbon_type",
            "wlbName": "discovery_wellbore",
            "nmaName": "main_area_name",
            "fldName": "associated_field",
            "dscDateFromInclInField": "field_inclusion_date",
            "dscDiscoveryYear": "discovery_year",
            "dscResInclInDiscoveryName": "parent_discovery",
            "dscOwnerKind": "owner_type",
            "dscOwnerName": "owner_name",
            "dscNpdidDiscovery": "find_id",
            "fldNpdidField": "field_id",
            "wlbNpdidWellbore": "wellbore_id",
            "dscFactPageUrl": "fact_page_url",
            "dscFactMapUrl": "fact_map_url",
            "dscDateUpdated": "date_updated",
            "dscDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Hydrocarbon discovery events on the Norwegian continental shelf",
    },
    "discovery_reserves": {
        "new_name": "find_resource_estimates",
        "columns": {
            "dscName": "find_name",
            "dscReservesRC": "resource_class",
            "dscRecoverableOil": "recoverable_oil",
            "dscRecoverableGas": "recoverable_gas",
            "dscRecoverableNGL": "recoverable_ngl",
            "dscRecoverableCondensate": "recoverable_condensate",
            "dscDateOffResEstDisplay": "estimate_display_date",
            "dscNpdidDiscovery": "find_id",
            "dscReservesDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Estimated recoverable resources for hydrocarbon discoveries",
    },
    "dscArea": {
        "new_name": "discovery_field_mapping",
        "columns": {
            "fldNpdidField": "field_id",
            "fldName": "field_name",
            "dscNpdidDiscovery": "discovery_id",
            "dscName": "discovery_name",
            "dscResInclInDiscoveryName": "parent_discovery_name",
            "dscNpdidResInclInDiscovery": "parent_discovery_id",
            "dscIncludedInFld": "included_in_field",
            "dscHcType": "hydrocarbon_type",
            "fldHcType": "field_hc_type",
            "dscCurrentActivityStatus": "discovery_status",
            "fldCurrentActivityStatus": "field_status",
            "flddscLabel": "display_label",
            "dscFactUrl": "discovery_fact_url",
            "fldFactUrl": "field_fact_url",
        },
        "description": "Mapping between discoveries and their associated petroleum fields",
    },

    # =========================================================================
    # NRG Dataset - Facilities
    # =========================================================================
    "facility": {
        "new_name": "offshore_installations",
        "columns": {
            "fclNpdidFacility": "installation_id",
            "fclName": "installation_name",
            "fclKind": "installation_type",
            "fclCurrentOperator": "current_operator_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Offshore petroleum production installations",
    },
    "facility_fixed": {
        "new_name": "permanent_platforms",
        "columns": {
            "fclName": "platform_name",
            "fclPhase": "lifecycle_phase",
            "fclSurface": "surface_type",
            "fclCurrentOperatorName": "operator_name",
            "fclKind": "platform_type",
            "fclBelongsToName": "parent_facility_name",
            "fclBelongsToKind": "parent_facility_type",
            "fclBelongsToS": "parent_facility_id",
            "fclStartupDate": "commissioning_date",
            "fclGeodeticDatum": "geodetic_datum",
            "fclNsDeg": "latitude_degrees",
            "fclNsMin": "latitude_minutes",
            "fclNsSec": "latitude_seconds",
            "fclNsCode": "latitude_hemisphere",
            "fclEwDeg": "longitude_degrees",
            "fclEwMin": "longitude_minutes",
            "fclEwSec": "longitude_seconds",
            "fclEwCode": "longitude_hemisphere",
            "fclWaterDepth": "water_depth_m",
            "fclFunctions": "platform_functions",
            "fclDesignLifetime": "design_lifetime_years",
            "fclFactPageUrl": "fact_page_url",
            "fclFactMapUrl": "fact_map_url",
            "fclNpdidFacility": "platform_id",
            "fclDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Permanently installed offshore platforms and structures",
    },
    "facility_moveable": {
        "new_name": "mobile_drilling_units",
        "columns": {
            "fclName": "unit_name",
            "fclCurrentRespCompanyName": "responsible_company",
            "fclKind": "unit_type",
            "fclFunctions": "unit_functions",
            "fclNationName": "flag_nation",
            "fclFactPageUrl": "fact_page_url",
            "fclNpdidFacility": "unit_id",
            "fclNpdidCurrentRespCompany": "responsible_company_id",
            "fclDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Mobile offshore drilling rigs and floating production units",
    },
    "fclPoint": {
        "new_name": "facility_locations",
        "columns": {
            "fclNpdidFacility": "facility_id",
            "fclSurface": "surface_type",
            "fclCurrentOperatorName": "operator_name",
            "fclName": "facility_name",
            "fclKind": "facility_type",
            "fclBelongsToName": "parent_name",
            "fclBelongsToKind": "parent_type",
            "fclBelongsToS": "parent_id",
            "fclStartupDate": "startup_date",
            "fclWaterDepth": "water_depth_m",
            "fclFunctions": "functions",
            "fclDesignLifetime": "design_lifetime_years",
            "fclFactPageUrl": "fact_page_url",
            "fclFactMapUrl": "fact_map_url",
        },
        "description": "Geographic point locations of offshore facilities",
    },

    # =========================================================================
    # NRG Dataset - Field details (activity, description, investment, etc.)
    # =========================================================================
    "field_activity_status_hst": {
        "new_name": "deposit_status_history",
        "columns": {
            "fldName": "deposit_name",
            "fldStatusFromDate": "status_from",
            "fldStatusToDate": "status_to",
            "fldStatus": "activity_status",
            "fldNpdidField": "deposit_id",
            "fldStatusDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical activity status changes for petroleum deposits",
    },
    "field_description": {
        "new_name": "deposit_narratives",
        "columns": {
            "fldName": "deposit_name",
            "fldDescriptionHeading": "section_heading",
            "fldDescriptionText": "narrative_text",
            "fldNpdidField": "deposit_id",
            "fldDescriptionDateUpdated": "date_updated",
        },
        "description": "Descriptive narrative texts about petroleum deposit fields",
    },
    "field_investment_yearly": {
        "new_name": "annual_capital_expenditure",
        "columns": {
            "prfInformationCarrier": "asset_name",
            "prfYear": "fiscal_year",
            "prfInvestmentsMillNOK": "investment_mnok",
            "prfNpdidInformationCarrier": "asset_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Annual capital investment figures for petroleum fields",
    },
    "field_licensee_hst": {
        "new_name": "deposit_ownership_history",
        "columns": {
            "fldName": "deposit_name",
            "fldOwnerName": "owner_name",
            "fldOwnerKind": "owner_type",
            "fldOwnerFrom": "ownership_from",
            "fldOwnerTo": "ownership_to",
            "fldLicenseeFrom": "licence_from",
            "fldLicenseeTo": "licence_to",
            "cmpLongName": "company_name",
            "fldCompanyShare": "ownership_share_pct",
            "fldSdfiShare": "state_share_pct",
            "fldNpdidField": "deposit_id",
            "cmpNpdidCompany": "company_id",
            "fldLicenseeDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical ownership and licence participation in deposits",
    },
    "field_owner_hst": {
        "new_name": "deposit_title_holders",
        "columns": {
            "fldName": "deposit_name",
            "fldOwnerKind": "holder_type",
            "fldOwnerName": "holder_name",
            "fldOwnershipFromDate": "held_from",
            "fldOwnershipToDate": "held_to",
            "fldNpdidField": "deposit_id",
            "fldNpdidOwner": "holder_id",
            "fldOwnerDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Title holder history for petroleum deposit fields",
    },
    "field_production_monthly": {
        "new_name": "monthly_extraction_volumes",
        "columns": {
            "prfInformationCarrier": "field_name",
            "prfYear": "year",
            "prfMonth": "month",
            "prfPrdOilNetMillSm3": "oil_msm3",
            "prfPrdGasNetBillSm3": "gas_bsm3",
            "prfPrdNGLNetMillSm3": "ngl_msm3",
            "prfPrdCondensateNetMillSm3": "condensate_msm3",
            "prfPrdOeNetMillSm3": "oil_equivalent_msm3",
            "prfPrdProducedWaterInFieldMillSm3": "produced_water_msm3",
            "prfNpdidInformationCarrier": "field_id",
        },
        "description": "Monthly petroleum extraction volumes by field",
    },
    "field_production_totalt_NCS_month": {
        "new_name": "ncs_monthly_aggregate_output",
        "columns": {
            "prfYear": "year",
            "prfMonth": "month",
            "prfPrdOilNetMillSm3": "total_oil_msm3",
            "prfPrdGasNetBillSm3": "total_gas_bsm3",
            "prfPrdNGLNetMillSm3": "total_ngl_msm3",
            "prfPrdCondensateNetMillSm3": "total_condensate_msm3",
            "prfPrdOeNetMillSm3": "total_oe_msm3",
            "prfPrdProducedWaterInFieldMillSm3": "total_water_msm3",
        },
        "description": "Aggregated monthly production totals for the Norwegian Continental Shelf",
    },
    "field_production_totalt_NCS_year": {
        "new_name": "ncs_annual_aggregate_output",
        "columns": {
            "prfYear": "year",
            "prfPrdOilNetMillSm": "total_oil_msm3",
            "prfPrdGasNetBillSm": "total_gas_bsm3",
            "prfPrdCondensateNetMillSm3": "total_condensate_msm3",
            "prfPrdNGLNetMillSm3": "total_ngl_msm3",
            "prfPrdOeNetMillSm3": "total_oe_msm3",
            "prfPrdProducedWaterInFieldMillSm3": "total_water_msm3",
        },
        "description": "Aggregated annual production totals for the Norwegian Continental Shelf",
    },
    "fldArea": {
        "new_name": "field_discovery_areas",
        "columns": {
            "fldNpdidField": "field_id",
            "fldName": "field_name",
            "dscNpdidDiscovery": "discovery_id",
            "dscName": "discovery_name",
            "dscResInclInDiscoveryName": "parent_discovery",
            "dscNpdidResInclInDiscovery": "parent_discovery_id",
            "dscIncludedInFld": "in_field",
            "dscHcType": "discovery_hc_type",
            "fldHcType": "field_hc_type",
            "dscCurrentActivityStatus": "discovery_status",
            "fldCurrentActivityStatus": "field_status",
            "flddscLabel": "label",
            "dscFactUrl": "discovery_url",
            "fldFactUrl": "field_url",
        },
        "description": "Area mapping between petroleum fields and their discoveries",
    },

    # =========================================================================
    # NRG Dataset - Licences
    # =========================================================================
    "licence": {
        "new_name": "exploration_permits",
        "columns": {
            "prlName": "permit_name",
            "prlLicensingActivityName": "licensing_round",
            "prlMainArea": "main_area",
            "prlStatus": "permit_status",
            "prlDateGranted": "grant_date",
            "prlDateValidTo": "expiry_date",
            "prlOriginalArea": "original_area_km2",
            "prlCurrentArea": "current_area",
            "prlPhaseCurrent": "current_phase",
            "prlNpdidLicence": "permit_id",
            "prlFactPageUrl": "fact_page_url",
            "prlFactMapUrl": "fact_map_url",
            "prlDateUpdated": "date_updated",
            "prlDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Petroleum exploration and production permits",
    },
    "licence_area_poly_hst": {
        "new_name": "permit_boundary_history",
        "columns": {
            "prlName": "permit_name",
            "prlAreaPolyDateValidFrom": "boundary_valid_from",
            "prlAreaPolyDateValidTo": "boundary_valid_to",
            "prlAreaPolyNationCode": "country_code",
            "prlAreaPolyBlockName": "block_name",
            "prlAreaPolyStratigraphical": "stratigraphic_layer",
            "prlAreaPolyPolyNo": "polygon_number",
            "prlAreaPolyPolyArea": "polygon_area_km2",
            "prlNpdidLicence": "permit_id",
            "prlAreaDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical boundary polygon changes for exploration permits",
    },
    "licence_licensee_hst": {
        "new_name": "permit_participant_history",
        "columns": {
            "prlName": "permit_name",
            "prlLicenseeDateValidFrom": "participation_from",
            "prlLicenseeDateValidTo": "participation_to",
            "cmpLongName": "company_name",
            "prlLicenseeInterest": "ownership_share",
            "prlLicenseeSdfi": "state_share",
            "prlOperDateValidFrom": "operator_from",
            "prlOperDateValidTo": "operator_to",
            "prlNpdidLicence": "permit_id",
            "cmpNpdidCompany": "company_id",
            "prlLicenseeDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical participation stakes in exploration permits",
    },
    "licence_oper_hst": {
        "new_name": "permit_operator_history",
        "columns": {
            "prlName": "permit_name",
            "prlOperDateValidFrom": "operator_from",
            "prlOperDateValidTo": "operator_to",
            "cmpLongName": "operator_company",
            "prlNpdidLicence": "permit_id",
            "cmpNpdidCompany": "company_id",
            "prlOperDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Historical operator assignments for exploration permits",
    },
    "licence_petreg_licence": {
        "new_name": "regulatory_concessions",
        "columns": {
            "ptlName": "concession_name",
            "ptlDateAwarded": "award_date",
            "ptlDateValidFrom": "valid_from",
            "ptlDateValidTo": "valid_to",
            "prlNpdidLicence": "permit_id",
            "ptlDateUpdated": "date_updated",
            "ptlDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Petroleum registry concession records",
    },
    "licence_petreg_licence_licencee": {
        "new_name": "concession_holders",
        "columns": {
            "ptlName": "concession_name",
            "cmpLongName": "holder_company",
            "ptlLicenseeInterest": "holder_share",
            "prlNpdidLicence": "permit_id",
            "cmpNpdidCompany": "company_id",
            "ptlLicenseeDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Companies holding stakes in petroleum concessions",
    },
    "licence_petreg_licence_oper": {
        "new_name": "concession_operators",
        "columns": {
            "ptlName": "concession_name",
            "cmpLongName": "operating_company",
            "prlNpdidLicence": "permit_id",
            "cmpNpdidCompany": "company_id",
            "ptlOperDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Designated operators for petroleum concessions",
    },
    "licence_petreg_message": {
        "new_name": "concession_notices",
        "columns": {
            "prlName": "permit_name",
            "ptlMessageDocumentNo": "document_number",
            "ptlMessage": "notice_text",
            "ptlMessageRegisteredDate": "registration_date",
            "ptlMessageKindDesc": "notice_type",
            "ptlMessageDateUpdated": "date_updated",
            "prlNpdidLicence": "permit_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Official notices and messages for petroleum concessions",
    },
    "licence_phase_hst": {
        "new_name": "permit_phase_transitions",
        "columns": {
            "prlName": "permit_name",
            "prlDatePhaseValidFrom": "phase_from",
            "prlDatePhaseValidTo": "phase_to",
            "prlPhase": "phase_name",
            "prlDateGranted": "grant_date",
            "prlDateValidTo": "permit_expiry",
            "prlDateInitialPeriodExpires": "initial_period_expiry",
            "prlActiveStatusIndicator": "is_active",
            "prlNpdidLicence": "permit_id",
            "prlPhaseDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Phase transition history for exploration permits",
    },
    "licence_task": {
        "new_name": "permit_work_obligations",
        "columns": {
            "prlName": "permit_name",
            "prlTaskName": "obligation_name",
            "prlTaskTypeNo": "task_type_no",
            "prlTaskTypeEn": "task_type_en",
            "prlTaskStatusNo": "task_status_no",
            "prlTaskStatusEn": "task_status_en",
            "prlTaskExpiryDate": "deadline",
            "wlbName": "wellbore_name",
            "prlDateValidTo": "permit_expiry",
            "prlLicensingActivityName": "licensing_round",
            "cmpLongName": "responsible_company",
            "cmpNpdidCompany": "company_id",
            "prlNpdidLicence": "permit_id",
            "prlTaskID": "task_id",
            "prlTaskRefID": "task_ref_id",
            "prlTaskDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Work programme obligations tied to exploration permits",
    },
    "licence_transfer_hst": {
        "new_name": "permit_stake_transfers",
        "columns": {
            "prlName": "permit_name",
            "prlTransferDateValidFrom": "transfer_date",
            "prlTransferDirection": "transfer_direction",
            "prlTransferKind": "transfer_type",
            "cmpLongName": "company_name",
            "prlTransferredInterest": "transferred_share",
            "prlTransferSdfi": "state_share",
            "prlNpdidLicence": "permit_id",
            "cmpNpdidCompany": "company_id",
            "prlTransferDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Ownership stake transfer history for exploration permits",
    },

    # =========================================================================
    # NRG Dataset - Pipelines, Production Licences, Areas
    # =========================================================================
    "pipLine": {
        "new_name": "subsea_pipelines",
        "columns": {
            "pipNpdidPipe": "pipeline_id",
            "pipNpdidFromFacility": "from_facility_id",
            "pipNpdidToFacility": "to_facility_id",
            "pipNpdidOperator": "operator_id",
            "pipName": "pipeline_name",
            "pipNameFromFacility": "origin_facility",
            "pipNameToFacility": "destination_facility",
            "pipNameCurrentOperator": "current_operator",
            "pipCurrentPhase": "lifecycle_phase",
            "pipMedium": "transport_medium",
            "pipMainGrouping": "system_group",
            "pipDimension": "diameter_inches",
        },
        "description": "Subsea pipeline connections between offshore facilities",
    },
    "prlArea": {
        "new_name": "permit_area_details",
        "columns": {
            "prlName": "permit_name",
            "prlActive": "is_active",
            "prlCurrentArea": "current_area",
            "prlDateGranted": "grant_date",
            "prlDateValidTo": "expiry_date",
            "prlAreaPolyDateValidFrom": "polygon_from",
            "prlAreaPolyDateValidTo": "polygon_to",
            "prlAreaPolyFromZvalue": "depth_from",
            "prlAreaPolyToZvalue": "depth_to",
            "prlAreaPolyVertLimEn": "vertical_limit_en",
            "prlAreaPolyVertLimNo": "vertical_limit_no",
            "prlStratigraphical": "stratigraphic",
            "prlAreaPolyStratigraphical": "polygon_stratigraphic",
            "prlNpdidLicence": "permit_id",
            "prlLastOperatorNameShort": "operator_short",
            "prlLastOperatorNameLong": "operator_long",
            "prlLicensingActivityName": "licensing_round",
            "prlLastOperatorNpdidCompany": "operator_id",
            "prlFactUrl": "fact_url",
            "prlArea_id": "area_record_id",
        },
        "description": "Detailed area specifications for petroleum exploration permits",
    },
    "prlAreaSplitByBlock": {
        "new_name": "permit_block_allocations",
        "columns": {
            "prlName": "permit_name",
            "prlActive": "is_active",
            "prlCurrentArea": "current_area",
            "prlDateGranted": "grant_date",
            "prlDateValidTo": "expiry_date",
            "prlAreaPolyDateValidFrom": "polygon_from",
            "prlAreaPolyDateValidTo": "polygon_to",
            "prlAreaPolyPolyNo": "polygon_number",
            "prlAreaPolyPolyArea": "polygon_area_km2",
            "blcName": "block_name",
            "prlAreaPolyFromZvalue": "depth_from",
            "prlAreaPolyToZvalue": "depth_to",
            "prlAreaPolyVertLimEn": "vertical_limit_en",
            "prlAreaPolyVertLimNo": "vertical_limit_no",
            "prlStratigraphical": "stratigraphic",
            "prlLastOperatorNpdidCompany": "operator_id",
            "prlLastOperatorNameShort": "operator_short",
            "prlLastOperatorNameLong": "operator_long",
            "prlLicensingActivityName": "licensing_round",
            "prlFactUrl": "fact_url",
            "prlAreaPolyStratigraphical": "polygon_stratigraphic",
            "prlNpdidLicence": "permit_id",
        },
        "description": "Permit areas broken down by individual exploration blocks",
    },
    "production_licence": {
        "new_name": "production_concessions",
        "columns": {
            "prlNpdidLicence": "concession_id",
            "prlName": "concession_name",
            "prlStatus": "status",
            "prlDateGranted": "grant_date",
            "prlDateValidTo": "expiry_date",
            "prlAreaSize": "area_size_km2",
            "dateSyncNPD": "sync_date",
        },
        "description": "Production licences for petroleum extraction operations",
    },

    # =========================================================================
    # NRG Dataset - Seismic surveys
    # =========================================================================
    "seaArea": {
        "new_name": "seismic_survey_areas",
        "columns": {
            "seaSurveyName": "survey_name",
            "seaNpdidSurvey": "survey_id",
            "seaFactMapUrl": "map_url",
            "seaFactPageUrl": "fact_url",
            "seaStatus": "survey_status",
            "seaGeographicalArea": "geographic_area",
            "seaMarketAvailable": "market_available",
            "seaSurveyTypeMain": "main_survey_type",
            "seaSurveyTypePart": "sub_survey_type",
            "seaCompanyReported": "reporting_company",
            "seaSourceType": "source_type",
            "seaSourceNumber": "source_count",
            "seaSourceSize": "source_size",
            "seaSourcePressure": "source_pressure",
            "seaSensorType": "sensor_type",
            "seaSensorNumbers": "sensor_count",
            "seaSensorLength": "sensor_length",
            "seaPlanFromDate": "planned_start",
            "seaDateStarting": "actual_start",
            "seaPlanToDate": "planned_end",
            "seaDateFinalized": "actual_end",
            "seaPlanCdpKm": "planned_cdp_km",
            "seaCdpTotalKm": "actual_cdp_km",
            "seaPlanBoatKm": "planned_boat_km",
            "seaBoatTotalKm": "actual_boat_km",
            "sea3DKm2": "coverage_3d_km2",
            "seaPolygonKind": "polygon_type",
            "seaArea_id": "area_record_id",
        },
        "description": "Seismic survey coverage areas and acquisition parameters",
    },
    "seaMultiline": {
        "new_name": "seismic_line_surveys",
        "columns": {
            "seaSurveyName": "survey_name",
            "seaFactMapUrl": "map_url",
            "seaFactPageUrl": "fact_url",
            "seaStatus": "survey_status",
            "seaMarketAvailable": "market_available",
            "seaSurveyTypeMain": "main_type",
            "seaSurveyTypePart": "sub_type",
            "seaCompanyReported": "reporting_company",
            "seaSourceType": "source_type",
            "seaSourceNumber": "source_count",
            "seaSourceSize": "source_size",
            "seaSourcePressure": "source_pressure",
            "seaSensorType": "sensor_type",
            "seaSensorNumbers": "sensor_count",
            "seaSensorLength": "sensor_length",
            "seaPlanFromDate": "planned_start",
            "seaDateStarting": "actual_start",
            "seaPlanToDate": "planned_end",
            "seaDateFinalized": "actual_end",
            "seaPlanCdpKm": "planned_cdp_km",
            "seaCdpTotalKm": "actual_cdp_km",
            "seaPlanBoatKm": "planned_boat_km",
            "seaBoatTotalKm": "actual_boat_km",
        },
        "description": "Multi-line seismic survey acquisition records",
    },
    "seis_acquisition": {
        "new_name": "geophysical_campaigns",
        "columns": {
            "seaName": "campaign_name",
            "seaPlanFromDate": "planned_start",
            "seaNpdidSurvey": "survey_id",
            "seaStatus": "campaign_status",
            "seaGeographicalArea": "geographic_area",
            "seaSurveyTypeMain": "main_type",
            "seaSurveyTypePart": "sub_type",
            "seaCompanyReported": "reporting_company",
            "seaPlanToDate": "planned_end",
            "seaDateStarting": "actual_start",
            "seaDateFinalized": "actual_end",
            "seaCdpTotalKm": "total_cdp_km",
            "seaBoatTotalKm": "total_vessel_km",
            "sea3DKm2": "coverage_3d_km2",
            "seaSampling": "sampling_info",
            "seaShallowDrilling": "shallow_drilling",
            "seaGeotechnical": "geotechnical_info",
            "dateSyncNPD": "sync_date",
        },
        "description": "Geophysical seismic acquisition campaign records",
    },
    "seis_acquisition_coordinates_inc_turnarea": {
        "new_name": "survey_polygon_coordinates",
        "columns": {
            "seaSurveyName": "survey_name",
            "seaNpdidSurvey": "survey_id",
            "seaPolygonPointNumber": "point_sequence",
            "seaPolygonNSDeg": "lat_degrees",
            "seaPolygonNSMin": "lat_minutes",
            "seaPolygonNSSec": "lat_seconds",
            "seaPolygonEWDeg": "lon_degrees",
            "seaPolygonEWMin": "lon_minutes",
            "seaPolygonEWSec": "lon_seconds",
            "dateSyncNPD": "sync_date",
        },
        "description": "Boundary polygon vertices for seismic survey coverage areas",
    },
    "seis_acquisition_progress": {
        "new_name": "survey_progress_reports",
        "columns": {
            "seaProgressDate": "report_date",
            "seaProgressText2": "summary_text",
            "seaProgressText": "detail_text",
            "seaProgressDescription": "description",
            "seaNpdidSurvey": "survey_id",
            "seis_acquisition_progress_id": "report_id",
        },
        "description": "Progress tracking reports for seismic acquisition campaigns",
    },

    # =========================================================================
    # NRG Dataset - Stratigraphy
    # =========================================================================
    "strat_litho_wellbore": {
        "new_name": "geological_layer_penetrations",
        "columns": {
            "wlbName": "wellbore_name",
            "lsuTopDepth": "layer_top_depth",
            "lsuBottomDepth": "layer_bottom_depth",
            "lsuName": "formation_name",
            "lsuLevel": "stratigraphic_level",
            "lsuNpdidLithoStrat": "formation_id",
            "wlbCompletionDate": "well_completion_date",
            "wlbNpdidWellbore": "wellbore_id",
            "lsuWellboreUpdatedDate": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Geological formation layers penetrated by wellbores",
    },
    "strat_litho_wellbore_core": {
        "new_name": "geological_core_samples",
        "columns": {
            "wlbName": "wellbore_name",
            "lsuCoreLenght": "core_length_m",
            "lsuName": "formation_name",
            "lsuLevel": "stratigraphic_level",
            "wlbCompletionDate": "well_completion_date",
            "lsuNpdidLithoStrat": "formation_id",
            "wlbNpdidWellbore": "wellbore_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Core samples extracted from geological formations in wellbores",
    },

    # =========================================================================
    # NRG Dataset - TUF (Tillatelse til Unitisering og Fellesbruk)
    # =========================================================================
    "tuf_operator_hst": {
        "new_name": "unitization_operator_history",
        "columns": {
            "tufName": "unitization_name",
            "cmpLongName": "operator_company",
            "tufOperDateValidFrom": "operator_from",
            "tufOperDateValidTo": "operator_to",
            "tufNpdidTuf": "unitization_id",
            "cmpNpdidCompany": "company_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Operator history for unitization agreements",
    },
    "tuf_owner_hst": {
        "new_name": "unitization_ownership_history",
        "columns": {
            "tufName": "unitization_name",
            "cmpLongName": "owner_company",
            "tufOwnerDateValidFrom": "ownership_from",
            "tufOwnerDateValidTo": "ownership_to",
            "tufOwnerShare": "ownership_share_pct",
            "tufNpdidTuf": "unitization_id",
            "cmpNpdidCompany": "company_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Ownership stake history for unitization agreements",
    },
    "tuf_petreg_licence": {
        "new_name": "unitization_concessions",
        "columns": {
            "ptlName": "concession_name",
            "tufName": "unitization_name",
            "ptlDateValidFrom": "valid_from",
            "ptlDateValidTo": "valid_to",
            "tufNpdidTuf": "unitization_id",
            "ptlDateUpdated": "date_updated",
            "ptlDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Petroleum concessions associated with unitization agreements",
    },
    "tuf_petreg_licence_licencee": {
        "new_name": "unitization_concession_holders",
        "columns": {
            "ptlName": "concession_name",
            "cmpLongName": "holder_company",
            "ptlLicenseeInterest": "holder_share",
            "tufName": "unitization_name",
            "tufNpdidTuf": "unitization_id",
            "cmpNpdidCompany": "company_id",
            "ptlLicenseeDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Company stakes in unitization-linked concessions",
    },
    "tuf_petreg_licence_oper": {
        "new_name": "unitization_concession_managers",
        "columns": {
            "Textbox42": "label_42",
            "Textbox2": "label_2",
            "ptlName": "concession_name",
            "cmpLongName": "managing_company",
            "tufName": "unitization_name",
            "tufNpdidTuf": "unitization_id",
            "cmpNpdidCompany": "company_id",
            "ptlOperDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Operating companies for unitization-linked concessions",
    },
    "tuf_petreg_message": {
        "new_name": "unitization_notices",
        "columns": {
            "ptlName": "concession_name",
            "ptlMessageDocumentNo": "document_number",
            "ptlMessage": "notice_text",
            "ptlMessageRegisteredDate": "registration_date",
            "ptlMessageKindDesc": "notice_type",
            "tufName": "unitization_name",
            "ptlMessageDateUpdated": "date_updated",
            "tufNpdidTuf": "unitization_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Official notices for unitization agreements",
    },

    # =========================================================================
    # NRG Dataset - Wellbore details
    # =========================================================================
    "wellbore": {
        "new_name": "drilling_records",
        "columns": {
            "wlbNpdidWellbore": "record_id",
            "wlbWellboreName": "borehole_name",
            "wlbWell": "well_name",
            "wlbWellType": "well_type",
            "wlbStatus": "status",
            "wlbContent": "content_found",
            "wlbPurpose": "purpose",
            "wlbEntryYear": "spud_year",
            "wlbCompletionYear": "completion_year",
            "wlbTotalDepth": "total_depth_m",
            "wlbWaterDepth": "water_depth_m",
            "dateSyncNPD": "sync_date",
        },
        "description": "Basic drilling records for petroleum wellbores",
    },
    "wellbore_casing_and_lot": {
        "new_name": "borehole_casing_records",
        "columns": {
            "wlbName": "borehole_name",
            "wlbCasingType": "casing_type",
            "wlbCasingDiameter": "casing_diameter",
            "wlbCasingDepth": "casing_depth_m",
            "wlbHoleDiameter": "hole_diameter",
            "wlbHoleDepth": "hole_depth_m",
            "wlbLotMudDencity": "leak_off_mud_density",
            "wlbNpdidWellbore": "borehole_id",
            "wlbCasingDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
            "wellbore_casing_and_lot_id": "record_id",
        },
        "description": "Casing installations and leak-off test data for boreholes",
    },
    "wellbore_coordinates": {
        "new_name": "borehole_locations",
        "columns": {
            "wlbWellboreName": "borehole_name",
            "wlbDrillingOperator": "drilling_company",
            "wlbProductionLicence": "permit",
            "wlbWellType": "well_type",
            "wlbPurposePlanned": "planned_purpose",
            "wlbContent": "content_found",
            "wlbEntryDate": "spud_date",
            "wlbCompletionDate": "completion_date",
            "wlbField": "field_name",
            "wlbMainArea": "main_area",
            "wlbGeodeticDatum": "geodetic_datum",
            "wlbNsDeg": "lat_degrees",
            "wlbNsMin": "lat_minutes",
            "wlbNsSec": "lat_seconds",
            "wlbNsCode": "lat_hemisphere",
            "wlbEwDeg": "lon_degrees",
            "wlbEwMin": "lon_minutes",
            "wlbEwSec": "lon_seconds",
            "wlbEwCode": "lon_hemisphere",
            "wlbNsDecDeg": "latitude_decimal",
            "wlbEwDesDeg": "longitude_decimal",
            "wlbNsUtm": "utm_northing",
            "wlbEwUtm": "utm_easting",
            "wlbUtmZone": "utm_zone",
            "wlbNpdidWellbore": "borehole_id",
            "dateSyncNPD": "sync_date",
        },
        "description": "Geographic coordinates and location data for wellbores",
    },
    "wellbore_core": {
        "new_name": "borehole_core_inventory",
        "columns": {
            "wlbName": "borehole_name",
            "wlbCoreNumber": "core_number",
            "wlbCoreIntervalTop": "interval_top_m",
            "wlbCoreIntervalBottom": "interval_bottom_m",
            "wlbCoreIntervalUom": "interval_unit",
            "wlbTotalCoreLength": "total_core_length_m",
            "wlbNumberOfCores": "core_count",
            "wlbCoreSampleAvailable": "samples_available",
            "wlbNpdidWellbore": "borehole_id",
            "wlbCoreDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
            "wellbore_core_id": "inventory_id",
        },
        "description": "Core sample inventory records from wellbore drilling",
    },
    "wellbore_core_photo": {
        "new_name": "core_sample_imagery",
        "columns": {
            "wlbName": "borehole_name",
            "wlbCoreNumber": "core_number",
            "wlbCorePhotoTitle": "image_title",
            "wlbCorePhotoImgUrl": "image_url",
            "wlbNpdidWellbore": "borehole_id",
            "wlbCorePhotoDateUpdated": "date_updated",
            "wellbore_core_photo_id": "image_id",
        },
        "description": "Photographic documentation of wellbore core samples",
    },
    "wellbore_document": {
        "new_name": "borehole_documentation",
        "columns": {
            "wlbName": "borehole_name",
            "wlbDocumentType": "document_type",
            "wlbDocumentName": "document_name",
            "wlbDocumentUrl": "document_url",
            "wlbDocumentFormat": "file_format",
            "wlbDocumentSize": "file_size_mb",
            "wlbNpdidWellbore": "borehole_id",
            "wlbDocumentDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
            "wellbore_document_id": "document_id",
        },
        "description": "Technical documents and reports associated with wellbores",
    },
    "wellbore_dst": {
        "new_name": "drill_stem_tests",
        "columns": {
            "wlbName": "borehole_name",
            "wlbDstTestNumber": "test_number",
            "wlbDstFromDepth": "test_from_depth_m",
            "wlbDstToDepth": "test_to_depth_m",
            "wlbDstChokeSize": "choke_size",
            "wlbDstFinShutInPress": "final_shut_in_pressure",
            "wlbDstFinFlowPress": "final_flow_pressure",
            "wlbDstBottomHolePress": "bottom_hole_pressure",
            "wlbDstOilProd": "oil_flow_rate",
            "wlbDstGasProd": "gas_flow_rate",
            "wlbDstOilDensity": "oil_density",
            "wlbDstGasDensity": "gas_density",
            "wlbDstGasOilRelation": "gas_oil_ratio",
            "wlbNpdidWellbore": "borehole_id",
            "wlbDstDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Drill stem test results measuring reservoir pressure and flow",
    },
    "wellbore_formation_top": {
        "new_name": "formation_top_picks",
        "columns": {
            "wlbName": "borehole_name",
            "lsuTopDepth": "top_depth_m",
            "lsuBottomDepth": "bottom_depth_m",
            "lsuName": "formation_name",
            "lsuLevel": "stratigraphic_level",
            "lsuNameParent": "parent_formation",
            "wlbNpdidWellbore": "borehole_id",
            "lsuNpdidLithoStrat": "formation_id",
            "lsuNpdidLithoStratParent": "parent_formation_id",
            "lsuWellboreUpdatedDate": "date_updated",
            "dateSyncNPD": "sync_date",
        },
        "description": "Formation top depth picks identified in wellbore logs",
    },
    "wellbore_mud": {
        "new_name": "drilling_fluid_records",
        "columns": {
            "wlbName": "borehole_name",
            "wlbMD": "measured_depth_m",
            "wlbMudWeightAtMD": "mud_weight",
            "wlbMudViscosityAtMD": "mud_viscosity",
            "wlbYieldPointAtMD": "yield_point",
            "wlbMudType": "fluid_type",
            "wlbMudDateMeasured": "measurement_date",
            "wlbNpdidWellbore": "borehole_id",
            "wlbMudDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
            "wellbore_mud_id": "record_id",
        },
        "description": "Drilling fluid property measurements at various depths",
    },
    "wellbore_npdid_overview": {
        "new_name": "borehole_registry_index",
        "columns": {
            "wlbWellboreName": "borehole_name",
            "wlbNpdidWellbore": "registry_id",
            "wlbWell": "well_name",
            "wlbWellType": "well_type",
            "dateSyncNPD": "sync_date",
        },
        "description": "Registry index of all wellbore identifiers",
    },
    "wellbore_oil_sample": {
        "new_name": "hydrocarbon_fluid_samples",
        "columns": {
            "wlbName": "borehole_name",
            "wlbOilSampleTestType": "test_type",
            "wlbOilSampleTestNumber": "test_number",
            "wlbOilSampleTopDepth": "sample_top_depth_m",
            "wlbOilSampleBottomDepth": "sample_bottom_depth_m",
            "wlbOilSampleFluidType": "fluid_type",
            "wlbOilSampleTestDate": "test_date",
            "wlbOilSampledateReceivedDate": "received_date",
            "wlbNpdidWellbore": "borehole_id",
            "wlbOilSampleDateUpdated": "date_updated",
            "dateSyncNPD": "sync_date",
            "wellbore_oil_sample_id": "sample_id",
        },
        "description": "Hydrocarbon fluid samples collected during wellbore testing",
    },
    "wellbore_shallow_all": {
        "new_name": "shallow_drilling_operations",
        "columns": {
            "wlbWellboreName": "borehole_name",
            "wlbNpdidWellbore": "borehole_id",
            "wlbWell": "well_name",
            "wlbDrillingOperator": "drilling_company",
            "wlbProductionLicence": "permit",
            "wlbDrillingFacility": "drilling_rig",
            "wlbEntryDate": "spud_date",
            "wlbCompletionDate": "completion_date",
            "wlbDrillPermit": "drill_permit_no",
            "wlbTotalDepth": "total_depth_m",
            "wlbWaterDepth": "water_depth_m",
            "wlbMainArea": "main_area",
            "wlbEntryYear": "spud_year",
            "wlbCompletionYear": "completion_year",
            "wlbSeismicLocation": "seismic_location",
            "wlbGeodeticDatum": "geodetic_datum",
            "wlbNsDeg": "lat_degrees",
            "wlbNsMin": "lat_minutes",
            "wlbNsSec": "lat_seconds",
            "wlbNsCode": "lat_hemisphere",
            "wlbEwDeg": "lon_degrees",
            "wlbEwMin": "lon_minutes",
            "wlbEwSec": "lon_seconds",
            "wlbEwCode": "lon_hemisphere",
            "wlbNsDecDeg": "latitude_decimal",
            "wlbEwDesDeg": "longitude_decimal",
            "wlbNsUtm": "utm_northing",
            "wlbEwUtm": "utm_easting",
            "wlbUtmZone": "utm_zone",
            "wlbNamePart1": "name_part_1",
            "wlbNamePart2": "name_part_2",
            "wlbNamePart3": "name_part_3",
            "wlbNamePart4": "name_part_4",
            "wlbNamePart5": "name_part_5",
            "wlbNamePart6": "name_part_6",
            "wlbDateUpdated": "date_updated",
            "wlbDateUpdatedMax": "date_updated_max",
            "dateSyncNPD": "sync_date",
        },
        "description": "Shallow stratigraphic and geotechnical drilling operations",
    },
    "wlbPoint": {
        "new_name": "wellbore_map_points",
        "columns": {
            "wlbNpdidWellbore": "point_id",
            "wlbWellName": "well_name",
            "wlbWellboreName": "borehole_name",
            "wlbField": "field_name",
            "wlbProductionLicence": "permit",
            "wlbWellType": "well_type",
            "wlbDrillingOperator": "drilling_company",
            "wlbMultilateral": "is_multilateral",
            "wlbDrillingFacility": "drilling_rig",
            "wlbProductionFacility": "production_facility",
            "wlbEntryDate": "spud_date",
            "wlbCompletionDate": "completion_date",
            "wlbContent": "content_found",
            "wlbStatus": "status",
            "wlbSymbol": "map_symbol",
            "wlbPurpose": "purpose",
            "wlbWaterDepth": "water_depth_m",
            "wlbFactPageUrl": "fact_page_url",
            "wlbFactMapUrl": "fact_map_url",
            "wlbDiscoveryWellbore": "is_discovery_well",
        },
        "description": "Map point features for wellbore geographic visualization",
    },
}


# =============================================================================
# FK DEPENDENCIES: child_table -> (parent_table, child_fk_col, parent_pk_col)
# Defines how child tables follow their parent's partition assignment.
# =============================================================================

FK_DEPENDENCIES: dict[str, tuple[str, str, str]] = {
    # === EDU (roots: undergraduateStudent, graduateStudent) ===
    # LUBM concentrates all faculty in departments 0-1, so splitting by department
    # or university leaves one half empty. Instead we split at the student level
    # and duplicate all reference tables (department, university, faculty, courses).
    "undergraduateStudentTakeCourse":  ("undergraduateStudent", "undergraduateStudentID", "nr"),
    "graduateStudentTakeCourse":       ("graduateStudent", "graduateStudentID", "nr"),
    "coAuthorOfPublication":           ("graduateStudent", "graduateStudentID", "nr"),

    # === TRN (root: ROUTES — 13 entries; AGENCY has only 1 row) ===
    "TRIPS":           ("ROUTES", "route_id", "route_id"),
    "STOP_TIMES":      ("TRIPS", "trip_id", "trip_id"),
    "FREQUENCIES":     ("TRIPS", "trip_id", "trip_id"),

    # === NRG (root: field) ===
    "field_production_yearly":   ("field", "prfNpdidInformationCarrier", "fldNpdidField"),
    "field_operator_hst":        ("field", "fldNpdidField", "fldNpdidField"),
    "field_reserves":            ("field", "fldNpdidField", "fldNpdidField"),
    "field_activity_status_hst": ("field", "fldNpdidField", "fldNpdidField"),
    "field_description":         ("field", "fldNpdidField", "fldNpdidField"),
    "field_investment_yearly":   ("field", "prfNpdidInformationCarrier", "fldNpdidField"),
    "field_licensee_hst":        ("field", "fldNpdidField", "fldNpdidField"),
    "field_owner_hst":           ("field", "fldNpdidField", "fldNpdidField"),
    "field_production_monthly":  ("field", "prfNpdidInformationCarrier", "fldNpdidField"),
    "company_reserves":          ("field", "fldNpdidField", "fldNpdidField"),
    "discovery":                 ("field", "fldNpdidField", "fldNpdidField"),
    "discovery_reserves":        ("discovery", "dscNpdidDiscovery", "dscNpdidDiscovery"),
    "fldArea":                   ("field", "fldNpdidField", "fldNpdidField"),
    "dscArea":                   ("field", "fldNpdidField", "fldNpdidField"),

    # === BSBM (root: product) ===
    "offer":                  ("product", "product", "nr"),
    "review":                 ("product", "product", "nr"),
    "productfeatureproduct":  ("product", "product", "nr"),
    "producttypeproduct":     ("product", "product", "nr"),
}

# Tables to DUPLICATE in both halves (not split, but copied with renamed schema).
# These are shared reference tables or aggregate data that both halves need.
DUPLICATE_TABLES: set[str] = {
    # EDU: LUBM data concentrates all faculty in 2 of 15 departments; duplicate all
    # reference tables so both halves can resolve student→junction→course→faculty→dept
    "university", "department", "faculty", "professor", "lecturer",
    "undergraduateCourse", "graduateCourse", "researchGroup",
    "publication",
    # TRN: only 1 agency, plus shared reference/metadata tables
    "AGENCY", "STOPS", "CALENDAR", "CALENDAR_DATES", "SHAPES", "FEED_INFO",
    # NRG: company operates across many fields; NCS aggregate data
    "company",
    "field_production_totalt_NCS_month",
    "field_production_totalt_NCS_year",
}


def _topological_sort_dependencies() -> list[str]:
    """Topological sort of FK_DEPENDENCIES so parents are processed before children."""
    # Build adjacency: parent -> [children]
    children_of: dict[str, list[str]] = {}
    for child, (parent, _, _) in FK_DEPENDENCIES.items():
        children_of.setdefault(parent, []).append(child)

    visited: set[str] = set()
    order: list[str] = []

    def visit(table: str):
        if table in visited:
            return
        visited.add(table)
        for child in children_of.get(table, []):
            visit(child)
        order.append(table)

    # Start from root tables (parents not themselves in FK_DEPENDENCIES)
    roots = {parent for parent, _, _ in FK_DEPENDENCIES.values()} - set(FK_DEPENDENCIES.keys())
    for root in sorted(roots):
        visit(root)
    # Also visit any remaining unvisited
    for child in FK_DEPENDENCIES:
        visit(child)

    # Reverse post-order = topological order (parents first)
    order.reverse()
    # Filter to only child tables (roots don't need partition cascading)
    return [t for t in order if t in FK_DEPENDENCIES]


def compute_partition_assignment(
    conn: sqlite3.Connection,
) -> dict[str, set[int] | str]:
    """Compute FK-preserving partition for all tables in RENAME_MAP.

    Returns dict mapping table_name to either:
      - set of rowids to move to the renamed table
      - "DUPLICATE" for tables that should be fully copied (not split)
      - None for tables using naive 50/50 split
    """
    assignment: dict[str, set[int] | str] = {}

    # Mark duplicate tables
    for table in DUPLICATE_TABLES:
        if table in RENAME_MAP:
            assignment[table] = "DUPLICATE"

    # Identify root tables: in RENAME_MAP, not a child in FK_DEPENDENCIES, not DUPLICATE
    root_tables = [
        t for t in RENAME_MAP
        if t not in FK_DEPENDENCIES and t not in DUPLICATE_TABLES
    ]

    # Also identify implicit roots: parent tables referenced by FK_DEPENDENCIES
    # that ARE in RENAME_MAP but not themselves children
    fk_parents = {parent for parent, _, _ in FK_DEPENDENCIES.values()}
    implicit_roots = [
        t for t in fk_parents
        if t in RENAME_MAP and t not in FK_DEPENDENCIES and t not in DUPLICATE_TABLES
    ]

    # Split root tables 50/50 by rowid and store their PK values for children
    pk_values_moved: dict[str, set] = {}  # table -> set of PK values in the moved half

    all_roots = set(root_tables) | set(implicit_roots)
    for table in sorted(all_roots):
        cur = conn.execute(f'SELECT rowid FROM "{table}" ORDER BY rowid')
        all_rowids = [r[0] for r in cur.fetchall()]
        half = len(all_rowids) // 2
        if half == 0 and len(all_rowids) > 0:
            half = 1
        moved_rowids = set(all_rowids[:half])
        assignment[table] = moved_rowids

        # Determine which PK values were moved (needed for children)
        # Find the PK column used by children
        pk_cols_needed = set()
        for child, (parent, _, parent_pk) in FK_DEPENDENCIES.items():
            if parent == table:
                pk_cols_needed.add(parent_pk)

        for pk_col in pk_cols_needed:
            if moved_rowids:
                placeholders = ",".join("?" * len(moved_rowids))
                cur = conn.execute(
                    f'SELECT DISTINCT "{pk_col}" FROM "{table}" '
                    f'WHERE rowid IN ({placeholders})',
                    list(moved_rowids),
                )
                pk_values_moved[(table, pk_col)] = {r[0] for r in cur.fetchall()}
            else:
                pk_values_moved[(table, pk_col)] = set()

    # Process children in topological order
    topo_order = _topological_sort_dependencies()
    for child_table in topo_order:
        if child_table in assignment:
            continue  # Already assigned (e.g., DUPLICATE)
        if child_table not in RENAME_MAP:
            continue  # Not in our rename map, skip

        parent_table, child_fk_col, parent_pk_col = FK_DEPENDENCIES[child_table]
        parent_moved_pks = pk_values_moved.get((parent_table, parent_pk_col), set())

        if not parent_moved_pks:
            # Parent had no moved PKs → move first half naively
            cur = conn.execute(f'SELECT rowid FROM "{child_table}" ORDER BY rowid')
            all_rowids = [r[0] for r in cur.fetchall()]
            half = len(all_rowids) // 2
            if half == 0 and len(all_rowids) > 0:
                half = 1
            assignment[child_table] = set(all_rowids[:half])
        else:
            # Move child rows whose FK points to a moved parent
            placeholders = ",".join("?" * len(parent_moved_pks))
            cur = conn.execute(
                f'SELECT rowid FROM "{child_table}" '
                f'WHERE "{child_fk_col}" IN ({placeholders})',
                list(parent_moved_pks),
            )
            moved_rowids = {r[0] for r in cur.fetchall()}
            assignment[child_table] = moved_rowids

        # Store this child's PK values for its own children
        child_pk_cols_needed = set()
        for grandchild, (gp, _, gp_pk) in FK_DEPENDENCIES.items():
            if gp == child_table:
                child_pk_cols_needed.add(gp_pk)

        moved_rowids = assignment[child_table]
        if isinstance(moved_rowids, set) and moved_rowids:
            for pk_col in child_pk_cols_needed:
                placeholders = ",".join("?" * len(moved_rowids))
                cur = conn.execute(
                    f'SELECT DISTINCT "{pk_col}" FROM "{child_table}" '
                    f'WHERE rowid IN ({placeholders})',
                    list(moved_rowids),
                )
                pk_values_moved[(child_table, pk_col)] = {r[0] for r in cur.fetchall()}
        else:
            for pk_col in child_pk_cols_needed:
                pk_values_moved[(child_table, pk_col)] = set()

    # Remaining tables in RENAME_MAP: naive 50/50 split (no FK concerns)
    for table in RENAME_MAP:
        if table not in assignment:
            assignment[table] = None  # Signal for naive split

    return assignment


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> list[tuple]:
    """Get column info: [(cid, name, type, notnull, dflt_value, pk), ...]"""
    cur = conn.execute(f'PRAGMA table_info("{table_name}")')
    return cur.fetchall()


def create_renamed_table(
    conn: sqlite3.Connection,
    original_table: str,
    new_table: str,
    column_map: dict[str, str],
    partial: bool = False,
    rowids_to_move: set[int] | None = None,
):
    """Create a new table with renamed columns and move rows.

    Args:
        rowids_to_move: If provided, move exactly these rowids. If None, fall
            back to naive 50/50 split by rowid order.
    """
    schema = get_table_schema(conn, original_table)
    if not schema:
        print(f"  WARNING: Table {original_table} not found, skipping")
        return 0

    original_cols = [col[1] for col in schema]
    col_types = {col[1]: col[2] for col in schema}

    # Build new column definitions
    new_col_defs = []
    new_col_names = []
    for col_name in original_cols:
        new_name = column_map.get(col_name, col_name)  # Keep unmapped cols as-is
        col_type = col_types[col_name]
        new_col_defs.append(f'"{new_name}" {col_type}')
        new_col_names.append(new_name)

    # Create new table
    create_sql = f'CREATE TABLE "{new_table}" ({", ".join(new_col_defs)})'
    conn.execute(create_sql)

    # Count rows
    cur = conn.execute(f'SELECT COUNT(*) FROM "{original_table}"')
    total_rows = cur.fetchone()[0]
    if total_rows == 0:
        print(f"  {original_table} -> {new_table}: 0 rows (empty table)")
        return 0

    cols_quoted = ", ".join(f'"{c}"' for c in original_cols)
    new_cols_quoted = ", ".join(f'"{c}"' for c in new_col_names)

    if rowids_to_move is not None:
        # FK-preserving split: move exactly the specified rowids
        if not rowids_to_move:
            print(f"  {original_table} -> {new_table}: 0 rows (no matching FK rows)")
            return 0
        # Use temp table for the rowid set
        conn.execute('CREATE TEMP TABLE _split_ids (rid INTEGER)')
        # Insert in batches of 500 to avoid SQL variable limits
        rid_list = list(rowids_to_move)
        for i in range(0, len(rid_list), 500):
            batch = rid_list[i:i+500]
            placeholders = ",".join(f"({r})" for r in batch)
            conn.execute(f'INSERT INTO _split_ids VALUES {placeholders}')
    else:
        # Naive 50/50 split by rowid order
        half = total_rows // 2
        if half == 0 and total_rows > 0:
            half = 1
        conn.execute(f'''
            CREATE TEMP TABLE _split_ids AS
            SELECT rowid AS rid FROM "{original_table}" ORDER BY rowid LIMIT {half}
        ''')

    # Insert into new table
    conn.execute(f'''
        INSERT INTO "{new_table}" ({new_cols_quoted})
        SELECT {cols_quoted} FROM "{original_table}"
        WHERE rowid IN (SELECT rid FROM _split_ids)
    ''')

    # Delete those rows from original
    conn.execute(f'''
        DELETE FROM "{original_table}"
        WHERE rowid IN (SELECT rid FROM _split_ids)
    ''')

    conn.execute('DROP TABLE _split_ids')

    # Count moved rows
    cur = conn.execute(f'SELECT COUNT(*) FROM "{new_table}"')
    moved = cur.fetchone()[0]
    cur = conn.execute(f'SELECT COUNT(*) FROM "{original_table}"')
    remaining = cur.fetchone()[0]

    print(f"  {original_table} ({total_rows}) -> {new_table} ({moved}) + original ({remaining})")
    return moved


def create_duplicated_table(
    conn: sqlite3.Connection,
    original_table: str,
    new_table: str,
    column_map: dict[str, str],
):
    """Create a renamed copy with ALL rows. Original table keeps all rows too."""
    schema = get_table_schema(conn, original_table)
    if not schema:
        print(f"  WARNING: Table {original_table} not found, skipping")
        return 0

    original_cols = [col[1] for col in schema]
    col_types = {col[1]: col[2] for col in schema}

    new_col_defs = []
    new_col_names = []
    for col_name in original_cols:
        new_name = column_map.get(col_name, col_name)
        col_type = col_types[col_name]
        new_col_defs.append(f'"{new_name}" {col_type}')
        new_col_names.append(new_name)

    conn.execute(f'CREATE TABLE "{new_table}" ({", ".join(new_col_defs)})')

    cols_quoted = ", ".join(f'"{c}"' for c in original_cols)
    new_cols_quoted = ", ".join(f'"{c}"' for c in new_col_names)

    conn.execute(f'''
        INSERT INTO "{new_table}" ({new_cols_quoted})
        SELECT {cols_quoted} FROM "{original_table}"
    ''')

    cur = conn.execute(f'SELECT COUNT(*) FROM "{new_table}"')
    copied = cur.fetchone()[0]
    print(f"  {original_table} -> {new_table}: {copied} rows (DUPLICATED, original kept)")
    return copied


def generate_description_csv(
    conn: sqlite3.Connection,
    original_table: str,
    new_table: str,
    column_map: dict[str, str],
    table_description: str,
    output_dir: Path,
):
    """Generate a database_description CSV for the renamed table."""
    schema = get_table_schema(conn, new_table)
    if not schema:
        return

    # Try to load original CSV for descriptions
    original_csv = SOURCE_DESC_DIR / f"{original_table}.csv"
    original_descriptions = {}
    if original_csv.exists():
        with open(original_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_name = row.get("original_column_name", "")
                original_descriptions[orig_name] = row

    # Reverse column map for lookup
    reverse_map = {v: k for k, v in column_map.items()}

    csv_path = output_dir / f"{new_table}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "original_column_name", "column_name",
            "column_description", "data_format", "value_description",
        ])
        for col_info in schema:
            new_col_name = col_info[1]
            col_type = col_info[2]
            original_col = reverse_map.get(new_col_name, new_col_name)

            # Get original description if available
            orig_desc = original_descriptions.get(original_col, {})
            col_description = orig_desc.get("column_description", "")
            value_description = orig_desc.get("value_description", "")

            # Get example values from the new table
            try:
                cur = conn.execute(
                    f'SELECT DISTINCT "{new_col_name}" FROM "{new_table}" '
                    f'WHERE "{new_col_name}" IS NOT NULL LIMIT 5'
                )
                examples = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                if examples and not value_description:
                    value_description = f"Example values: {', '.join(examples[:4])}"
            except Exception:
                pass

            # Human-readable column name
            readable_name = new_col_name.replace("_", " ")

            writer.writerow([
                new_col_name, readable_name, col_description,
                col_type, value_description,
            ])


def verify_join_integrity(conn: sqlite3.Connection):
    """Verify that key JOINs return >0 rows in both halves (original + renamed)."""
    print(f"\n{'=' * 70}")
    print("FK Join Integrity Verification:")
    print(f"{'=' * 70}")

    checks = [
        # (description, original_sql, renamed_sql)
        (
            "EDU: student-course junction",
            'SELECT COUNT(*) FROM "undergraduateStudent" s JOIN "undergraduateStudentTakeCourse" j ON s.nr = j."undergraduateStudentID"',
            'SELECT COUNT(*) FROM "bachelor_participants" s JOIN "bachelor_enrollments" j ON s.participant_id = j.participant_id',
        ),
        (
            "EDU: faculty-department",
            'SELECT COUNT(*) FROM "faculty" f JOIN "department" d ON f."worksFor" = d.nr',
            'SELECT COUNT(*) FROM "academic_staff" f JOIN "organizational_units" d ON f.dept_id = d.unit_id',
        ),
        (
            "TRN: route-trip",
            'SELECT COUNT(*) FROM "ROUTES" r JOIN "TRIPS" t ON r.route_id = t.route_id',
            'SELECT COUNT(*) FROM "transit_lines" r JOIN "transit_journeys" t ON r.line_id = t.line_id',
        ),
        (
            "TRN: trip-stoptime-stop",
            'SELECT COUNT(*) FROM "TRIPS" t JOIN "STOP_TIMES" st ON t.trip_id = st.trip_id JOIN "STOPS" s ON st.stop_id = s.stop_id',
            'SELECT COUNT(*) FROM "transit_journeys" t JOIN "schedule_events" st ON t.journey_id = st.journey_id JOIN "transit_stations" s ON st.station_id = s.station_id',
        ),
        (
            "NRG: field-production",
            'SELECT COUNT(*) FROM "field" f JOIN "field_production_yearly" p ON f."fldNpdidField" = p."prfNpdidInformationCarrier"',
            'SELECT COUNT(*) FROM "petroleum_deposits" f JOIN "annual_output" p ON f.deposit_id = p.field_id',
        ),
        (
            "BSBM: product-offer",
            'SELECT COUNT(*) FROM "product" p JOIN "offer" o ON p.nr = o.product',
            'SELECT COUNT(*) FROM "merchandise" p JOIN "price_listings" o ON p.item_id = o.item_id',
        ),
    ]

    all_ok = True
    for desc, orig_sql, renamed_sql in checks:
        try:
            orig_count = conn.execute(orig_sql).fetchone()[0]
            renamed_count = conn.execute(renamed_sql).fetchone()[0]
            status = "OK" if orig_count > 0 and renamed_count > 0 else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  [{status}] {desc}: original={orig_count:,}, renamed={renamed_count:,}")
        except Exception as e:
            print(f"  [ERR] {desc}: {e}")
            all_ok = False

    if all_ok:
        print("\n  All FK join checks PASSED.")
    else:
        print("\n  WARNING: Some FK join checks FAILED!")

    return all_ok


def main():
    print("=" * 70)
    print("Creating heterogeneous CHESS database (FK-preserving split)")
    print("=" * 70)

    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DESC_DIR.mkdir(parents=True, exist_ok=True)

    # Copy source database
    print(f"\nCopying {SOURCE_DB.name} -> {TARGET_DB.name}...")
    shutil.copy2(SOURCE_DB, TARGET_DB)

    # Copy all existing description CSVs first
    print(f"Copying {SOURCE_DESC_DIR.name}/ description CSVs...")
    for csv_file in SOURCE_DESC_DIR.glob("*.csv"):
        shutil.copy2(csv_file, TARGET_DESC_DIR / csv_file.name)

    conn = sqlite3.connect(str(TARGET_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    # Compute FK-preserving partition assignment
    print("\nComputing FK-preserving partition assignment...")
    partition = compute_partition_assignment(conn)

    fk_split = sum(1 for v in partition.values() if isinstance(v, set))
    dup_count = sum(1 for v in partition.values() if v == "DUPLICATE")
    naive_count = sum(1 for v in partition.values() if v is None)
    print(f"  FK-cascade split: {fk_split} tables")
    print(f"  Duplicated: {dup_count} tables")
    print(f"  Naive split: {naive_count} tables")

    total_tables = 0
    total_rows_moved = 0
    total_rows_duplicated = 0

    print(f"\nProcessing {len(RENAME_MAP)} tables:\n")

    for original_table, config in RENAME_MAP.items():
        new_table = config["new_name"]
        column_map = config["columns"]
        description = config.get("description", "")
        partial = config.get("partial_columns", False)

        table_partition = partition.get(original_table)

        if table_partition == "DUPLICATE":
            copied = create_duplicated_table(
                conn, original_table, new_table, column_map
            )
            if copied > 0:
                total_tables += 1
                total_rows_duplicated += copied
        else:
            rowids = table_partition if isinstance(table_partition, set) else None
            moved = create_renamed_table(
                conn, original_table, new_table, column_map, partial,
                rowids_to_move=rowids,
            )
            if moved > 0:
                total_tables += 1
                total_rows_moved += moved

        generate_description_csv(
            conn, original_table, new_table, column_map,
            description, TARGET_DESC_DIR,
        )

    conn.commit()

    new_table_names = {v["new_name"] for v in RENAME_MAP.values()}

    # Basic stats
    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"{'=' * 70}")

    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [r[0] for r in cur.fetchall()]
    print(f"  Total tables: {len(all_tables)}")
    print(f"  Tables processed: {total_tables}")
    print(f"  Rows moved (split): {total_rows_moved:,}")
    print(f"  Rows duplicated: {total_rows_duplicated:,}")

    skip = {"sqlite_sequence"}
    original_tables = [t for t in all_tables if t not in new_table_names and t not in skip]
    new_tables = [t for t in all_tables if t in new_table_names]
    print(f"  Original tables: {len(original_tables)}")
    print(f"  Renamed copies: {len(new_tables)}")

    desc_files = list(TARGET_DESC_DIR.glob("*.csv"))
    print(f"  Description CSVs: {len(desc_files)} total")

    # Verify FK join integrity
    verify_join_integrity(conn)

    conn.close()

    print(f"\n{'=' * 70}")
    print(f"Output: {TARGET_DB}")
    print(f"Descriptions: {TARGET_DESC_DIR}")
    print(f"{'=' * 70}")
    print(f"\nTo run CHESS on this database, you need to:")
    print(f"1. Create dev.json with db_id='merged_heterogeneous'")
    print(f"2. Preprocess: python scripts/run_chess_experiment.py --preprocess-only")
    print(f"3. Run: python scripts/run_chess_experiment.py --models deepseek --merged")


if __name__ == "__main__":
    main()