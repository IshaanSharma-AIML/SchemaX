const Label = (props) => {
    const { name, htmlFor } = props;
    return (
        <label htmlFor={htmlFor} className="block text-sm font-medium text-sky-400 dark:text-gray-300">
            {name}
        </label>
    )
}
export default Label;