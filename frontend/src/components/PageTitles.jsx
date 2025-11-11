const PageTitles = (props) => {
    const { name } = props;
    return (
        <h1 className="text-3xl font-bold text-sky-400 mb-2">
            {name}
        </h1>
    )
}

export default PageTitles;